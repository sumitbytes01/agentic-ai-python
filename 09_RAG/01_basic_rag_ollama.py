# pip install -qU langchain-community pypdf
# pip install -qU langchain-text-splitters 
# pip install langchain-ollama


from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Change here: Use Ollama embeddings import instead of Google
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

from dotenv import load_dotenv
load_dotenv()

# load pdf
pdf_path = Path("/Users/sumit-macbookair/Study/ebooks/SystemDesign.pdf")

loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# chunking the text into smaller parts
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 10,
    chunk_overlap = 4
)

chunks = text_splitter.split_documents(
    documents=docs)

# Use your local Ollama embeddings model name here
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",   # or "mxbai-embed-large"
    base_url="http://localhost:11434"
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks, 
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name = "learning_rag",
    force_recreate=True
)

print("indexing of documentations done")

