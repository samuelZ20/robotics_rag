import pypdf
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

#lendo as paginas do pdf e conferindo se estão sendo lidas corretamente
def load_pdf(file_path: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata = {"source": file_path.name, "robot" : file_path.parent.name, "page_number": i + 1},
        )

        for i, page in enumerate(reader.pages)
    ]

docs = []

for file_path in Path("data").glob("**/*.pdf"):
    docs.extend(load_pdf(file_path))

#print(len(docs))

#dividir em chunks
def split_docs(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return splitter.split_documents(docs)

    chunks = splitter.split_documents(docs)
    print(len(docs), "páginas ->", len(chunks), "chunks")


