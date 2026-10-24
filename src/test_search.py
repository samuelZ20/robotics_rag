from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

pergunta = "qual a velocidade máxima do Misty?"

# menor score = mais parecido
for doc, score in db.similarity_search_with_score(pergunta, k=3):
    print(f"[{score:.3f}] {doc.metadata['robot']} | {doc.metadata['source']} | pág. {doc.metadata['page_number']}")
    print(doc.page_content[:200].replace("\n", " "), "\n")

# só um robô
print("--- só Go2 ---")
for doc in db.similarity_search(pergunta, k=3, filter={"robot": "go2"}):
    print(doc.metadata["source"], doc.metadata["page_number"])