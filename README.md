# robotics_rag

Assistente de perguntas e respostas (RAG) sobre a documentação oficial dos robôs do núcleo de robótica:

- **Unitree Go2** (`data/go2/`)
- **Misty II** (`data/misty/`)
- **Alpha Mini** (`data/alpha_mini/`)

A busca é feita localmente (embeddings + Chroma) e a resposta é gerada por um LLM — o **Gemini** (padrão) ou um modelo local via **Ollama** — usando só os trechos encontrados. As respostas saem em português e citam a fonte (robô, arquivo, página).

## Como funciona

1. **Ingestão** (`src/ingest_pdf.py`): lê os PDFs de `data/`, remove o ruído de impressão do navegador (cabeçalhos, rodapés, numeração, trilhas de navegação), divide o texto em chunks de 500 caracteres (overlap de 80) e salva os vetores em `chroma_db/`. Também avisa sobre PDFs possivelmente duplicados.
2. **Busca** (`src/rag_pipeline.py`): encontra os 4 chunks mais parecidos com a pergunta, opcionalmente filtrando por robô.
3. **Geração**: monta o prompt com esses trechos e chama o LLM. Erros da API (sem chave, chave inválida, limite 429, sem internet) viram avisos amigáveis, e os trechos encontrados continuam sendo mostrados.

Modelo de embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilíngue, então perguntas em português encontram trechos em inglês).

## Estrutura

```
data/                 PDFs da documentação, uma pasta por robô
src/
  ingest_pdf.py       gera o banco vetorial a partir dos PDFs
  rag_pipeline.py     busca + chamada ao LLM (usado pelo app e pela CLI)
  app.py              interface de chat em Streamlit
  query.py            pergunta pela linha de comando
  test_search.py      teste rápido só da busca (sem LLM)
chroma_db/            banco vetorial gerado (fora do git)
.env.example          modelo das variáveis de ambiente
```

## Instalação

Requer Python 3.10+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuração

Copie `.env.example` para `.env` na raiz do projeto e preencha:

```env
GOOGLE_API_KEY=sua_chave_aqui

# opcionais
# GEMINI_MODEL=gemini-3.5-flash-lite
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.2:1b
```

| Variável         | Padrão                  | Descrição                                  |
|------------------|-------------------------|--------------------------------------------|
| `GOOGLE_API_KEY` | —                       | Chave do Gemini (obrigatória para `gemini`) |
| `LLM_PROVIDER`   | `gemini`                | `gemini` ou `ollama`                       |
| `GEMINI_MODEL`   | `gemini-3.5-flash-lite` | Modelo do Gemini                           |
| `OLLAMA_MODEL`   | `llama3.2:1b`           | Modelo do Ollama                           |

A chave do Gemini é gerada em <https://aistudio.google.com> ("Get API key"). O `.env` está no `.gitignore` e nunca deve ser versionado.

Para usar o **Ollama**, instale-o, baixe o modelo (`ollama pull llama3.2:1b`), deixe o servidor rodando (`ollama serve`) e defina `LLM_PROVIDER=ollama`. Nesse caso a `GOOGLE_API_KEY` não é necessária.

## Uso

Todos os comandos são rodados a partir da **raiz do projeto** (os caminhos `data/` e `chroma_db/` são relativos a ela).

**1. Gerar o banco vetorial** (rodar de novo sempre que os PDFs mudarem; o banco anterior é apagado):

```bash
python src/ingest_pdf.py
```

Na primeira execução o modelo de embeddings é baixado do Hugging Face.

**2a. Interface web:**

```bash
streamlit run src/app.py
```

Na barra lateral dá para escolher o robô (ou "Todos") e limpar a conversa. Cada resposta mostra os trechos usados como fonte.

**2b. Linha de comando:**

```bash
python src/query.py "como desativar o desvio de obstáculos?"
python src/query.py "como fazer o robô dançar?" --robo alpha_mini
```

Valores de `--robo`: `go2`, `misty`, `alpha_mini`.

**Testar só a busca** (sem LLM):

```bash
python src/test_search.py
```

## Adicionando documentação

Coloque os PDFs em `data/<robo>/` e rode `python src/ingest_pdf.py` de novo. Para um robô novo, crie a pasta e adicione a entrada correspondente no dicionário `ROBOS` em `src/rag_pipeline.py` (nome da pasta → nome exibido).
