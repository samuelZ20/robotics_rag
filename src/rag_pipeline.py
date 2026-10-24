import os
import time
from functools import lru_cache

import httpx
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# mesmo modelo e mesma pasta do ingest_pdf.py (se o modelo mudar, os vetores não batem)
from ingest_pdf import EMBEDDING_MODEL, CHROMA_DIR


# lê o arquivo .env da raiz do projeto e coloca GOOGLE_API_KEY etc. nas variáveis de ambiente
load_dotenv()

# tudo configurável pelo .env, sem mexer no código
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()  # "gemini" (padrão) ou "ollama"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
K = 4                # quantos chunks entram no contexto
ESPERA_429 = 10      # segundos de espera antes de tentar de novo quando a API diz "limite atingido"

# nome da pasta em data/ -> nome bonito para mostrar
ROBOS = {"go2": "Unitree Go2", "misty": "Misty II", "alpha_mini": "Alpha Mini"}

MSG_SEM_CHAVE = (
    "A variável GOOGLE_API_KEY não está definida.\n"
    "1. Acesse https://aistudio.google.com, entre com sua conta Google e clique em \"Get API key\".\n"
    "2. Crie uma chave e copie.\n"
    "3. Na raiz do projeto, copie o arquivo .env.example para .env e preencha:\n"
    "   GOOGLE_API_KEY=sua_chave_aqui\n"
    "4. Rode o programa de novo. (O .env está no .gitignore, então a chave não vai para o git.)"
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um assistente do núcleo de robótica que responde dúvidas sobre os robôs "
     "Unitree Go2, Misty II e Alpha Mini.\n"
     "Regras:\n"
     "1. Responda SOMENTE com base nos trechos de documentação do CONTEXTO abaixo.\n"
     "2. Responda sempre em português, mesmo que os trechos estejam em inglês.\n"
     "3. Cite as fontes no formato (robô, arquivo, pág. N).\n"
     "4. Se o contexto não tiver a resposta, diga exatamente: "
     "\"Não encontrei isso na documentação.\" Não invente nada.\n\n"
     "CONTEXTO:\n{contexto}"),
    ("human", "{pergunta}"),
])


class ErroLLM(Exception):
    """Erro do LLM já traduzido numa mensagem amigável em português."""


#carrega o banco vetorial uma vez só (o modelo de embeddings demora para abrir)
@lru_cache(maxsize=1)
def abrir_banco() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


#escolhe o LLM conforme LLM_PROVIDER; criado uma vez só e reaproveitado
@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    if LLM_PROVIDER == "ollama":
        # importado só aqui: quem usa o Gemini não precisa ter o Ollama instalado
        from langchain_ollama import ChatOllama
        # num_ctx = janela de contexto; 2048 tokens cabem os 4 chunks e gastam menos RAM
        return ChatOllama(model=OLLAMA_MODEL, temperature=0, num_ctx=2048)

    if LLM_PROVIDER != "gemini":
        raise ErroLLM(f"LLM_PROVIDER inválido: '{LLM_PROVIDER}'. Use 'gemini' ou 'ollama'.")
    if not os.environ.get("GOOGLE_API_KEY"):
        raise ErroLLM(MSG_SEM_CHAVE)

    from langchain_google_genai import ChatGoogleGenerativeAI
    # temperature=0 = respostas mais fiéis ao contexto e menos "criativas"
    # max_retries=0: quem decide quando tentar de novo é chamar_llm() (só no erro 429)
    # a chave é lida sozinha da variável GOOGLE_API_KEY, então ela nunca aparece no código
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0, max_retries=0, timeout=60)


#chama o LLM e transforma os erros da API em mensagens que dá para mostrar ao usuário
def chamar_llm(mensagens) -> str:
    llm = get_llm()
    for tentativa in (1, 2):
        try:
            return llm.invoke(mensagens).text
        except ModelRateLimitError:
            if tentativa == 1:
                time.sleep(ESPERA_429)  # limite por minuto do plano gratuito: espera e tenta 1x
                continue
            raise ErroLLM("Limite de requisições da API do Gemini atingido (erro 429). "
                          "Espere um minuto e tente de novo; se persistir, a cota diária acabou.")
        except (ModelAuthenticationError, ModelPermissionDeniedError):
            raise ErroLLM("A chave GOOGLE_API_KEY foi recusada pelo Google. Confira o valor no .env.")
        except ModelInvalidRequestError as e:
            # o Gemini responde 400 "API key not valid" quando a chave está errada
            if "API key" in str(e):
                raise ErroLLM("A chave GOOGLE_API_KEY é inválida. Gere outra em https://aistudio.google.com e atualize o .env.")
            raise ErroLLM(f"A API do Gemini recusou a requisição: {e}")
        except ModelNotFoundError:
            raise ErroLLM(f"O modelo '{GEMINI_MODEL}' não existe ou não está disponível. "
                          "Troque GEMINI_MODEL no .env (veja https://ai.google.dev/gemini-api/docs/models).")
        except (httpx.TransportError, ConnectionError):
            if LLM_PROVIDER == "ollama":
                raise ErroLLM("Não consegui falar com o Ollama. Ele está aberto (ollama serve)?")
            raise ErroLLM("Sem conexão com a API do Gemini. Verifique sua internet.")


#busca os K chunks mais parecidos com a pergunta (opcionalmente só de um robô)
def buscar(pergunta: str, robo_filtro: str | None = None) -> list[tuple[Document, float]]:
    filtro = {"robot": robo_filtro} if robo_filtro else None
    return abrir_banco().similarity_search_with_score(pergunta, k=K, filter=filtro)


#junta os chunks num texto só, cada um com a sua etiqueta de fonte
def formatar_contexto(docs: list[Document]) -> str:
    partes = []
    for d in docs:
        m = d.metadata
        partes.append(f"[{ROBOS.get(m['robot'], m['robot'])}, {m['source']}, pág. {m['page_number']}]\n{d.page_content}")
    return "\n\n---\n\n".join(partes)


def gerar_resposta(pergunta: str, robo_filtro: str | None = None) -> tuple[str, list[tuple[Document, float]]]:
    """Retorna (resposta do LLM, lista de (chunk, score) usados como fonte).

    Se o LLM falhar, a resposta vira um aviso em português e as fontes da busca
    continuam sendo devolvidas, para o usuário ao menos ver os trechos encontrados.
    """
    fontes = buscar(pergunta, robo_filtro)
    if not fontes:
        return "Não encontrei isso na documentação.", []

    # só a pergunta e os trechos da documentação vão para a API (nada mais do usuário)
    mensagens = PROMPT.format_messages(
        contexto=formatar_contexto([d for d, _ in fontes]),
        pergunta=pergunta,
    )
    try:
        return chamar_llm(mensagens), fontes
    except ErroLLM as e:
        return f"⚠️ {e}", fontes
