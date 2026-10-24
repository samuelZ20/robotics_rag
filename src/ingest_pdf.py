import pypdf
import re
import shutil

from collections import Counter, defaultdict
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_DIR = "chroma_db"


# linhas que o navegador coloca em toda página impressa (não são conteúdo técnico)
PADROES_RUIDO = [
    re.compile(r"^\d{2}/\d{2}/\d{4}, \d{2}:\d{2} .+$"),  # cabeçalho: "03/10/2026, 23:21 Misty Docs | ..."
    re.compile(r"^https?://\S+ \d+/\d+$"),               # rodapé: "https://... 2/7"
    re.compile(r"^\d+/\d+$"),                            # numeração solta: "2/7"
    re.compile(r"^Search \.\.\. ⌘ K$"),                  # caixa de busca do site
    re.compile(r"^[-\s]+$"),                 # só ícones (caracteres de uso privado da fonte)
    re.compile(r"^[-]\s*».*$"),              # trilha de navegação: "  » Demos-Table of Contents » ..."
]


def eh_ruido(linha: str) -> bool:
    return any(p.match(linha.strip()) for p in PADROES_RUIDO)


# Obs.: não removemos "linhas que se repetem em muitas páginas" de forma genérica,
# porque nesses PDFs isso apagaria conteúdo técnico ("{", "Parameters", "Bases: ...BaseApi").
# Os padrões acima já pegam todo o ruído de impressão.
def limpar_texto(texto: str) -> str:
    return "\n".join(l for l in texto.splitlines() if not eh_ruido(l)).strip()


#lendo as paginas do pdf, limpando o ruído de impressão e guardando os metadados
def load_pdf(file_path: Path) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    textos = [limpar_texto(page.extract_text() or "") for page in reader.pages]
    return [
        Document(
            page_content=texto,
            metadata = {"source": file_path.name, "robot" : file_path.parent.name, "page_number": i + 1},
        )
        for i, texto in enumerate(textos)
        if texto  # pula páginas que ficaram vazias depois da limpeza
    ]

#dividir em chunks
def split_docs(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        add_start_index=True,
    )
    return splitter.split_documents(docs)

def build_vectorstore(chunks: list[Document]) -> Chroma:
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)  # evita chunks duplicados ao rodar de novo
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )


#procura PDFs duplicados comparando o texto (já limpo) das páginas; só avisa, não apaga nada
def encontrar_duplicatas(docs: list[Document], limite: float = 0.9) -> list[tuple[str, str, float]]:
    paginas_por_arquivo = defaultdict(set)
    for d in docs:
        paginas_por_arquivo[f"{d.metadata['robot']}/{d.metadata['source']}"].add(d.page_content)

    arquivos = sorted(paginas_por_arquivo)
    duplicatas = []
    for i, a in enumerate(arquivos):
        for b in arquivos[i + 1:]:
            pa, pb = paginas_por_arquivo[a], paginas_por_arquivo[b]
            # fração de páginas iguais em relação ao arquivo menor
            iguais = len(pa & pb) / min(len(pa), len(pb))
            if iguais >= limite:
                duplicatas.append((a, b, iguais))
    return duplicatas


if __name__ == "__main__":
    docs = []
    for file_path in Path("data").glob("**/*.pdf"):
        docs.extend(load_pdf(file_path))

    for a, b, iguais in encontrar_duplicatas(docs):
        print(f"AVISO: possível duplicata ({iguais:.0%} das páginas iguais): {a}  <->  {b}")

    chunks = split_docs(docs)
    print(len(docs), "páginas ->", len(chunks), "chunks")

    # resumo por robô
    paginas_robo = Counter(d.metadata["robot"] for d in docs)
    chunks_robo = Counter(c.metadata["robot"] for c in chunks)
    for robo in sorted(paginas_robo):
        print(f"  {robo:<11} {paginas_robo[robo]:>4} páginas -> {chunks_robo[robo]:>5} chunks")

    build_vectorstore(chunks)
    print("Chroma salvo em", CHROMA_DIR)
