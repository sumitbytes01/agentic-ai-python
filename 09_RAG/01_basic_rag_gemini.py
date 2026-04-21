# pip install -qU langchain-community pypdf
# pip install -qU langchain-text-splitters 
# pip install -qU langchain-google-genai

# 429
# but the code is fine

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from dotenv import load_dotenv
load_dotenv()

pdf_path = Path("/Users/sumit-macbookair/Study/ebooks/SystemDesign.pdf")

loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()
print(f"number of pages in document: {len(docs)}")
#print(docs[44])

# splitting text
# chunking the text into smaller parts
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 40
)

chunks = text_splitter.split_documents(
            documents=docs)

# create vector embedding for chunks
embedding_model = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-2-preview")


vector_store = QdrantVectorStore.from_documents(
    documents=chunks, 
    embedding=embedding_model,
    url="http://localhost:6333", # url of qdrant vector DB
    collection_name = "learning_rag",
    force_recreate = True
)

print("indexing of documentations done")

