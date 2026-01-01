from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

# load vector embedding
embedding_model = OllamaEmbeddings(
    model="lnomic-embed-text"
)

# Qdrant vector DB
vector_db = QdrantVectorStore.from_existing_collection(
    url="localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model
)

def process_query(user_query: str):
    print("searching chunks", user_query)
    search_result = vector_db.similarity_search(user_query)
    context = "\n\n\n".join([f"""Page Content:{result.page_content}\n
                        Page Number: {result.metadata['page_label']}\n
                        File Location: {result.metadata['source']}"""
                        for result in search_result])

    SYSTEM_PROMPT = f"""
                    You are an AI assistant who answers user query based on the available content 
                    retrieved from the PDF file 
                    along with page_content and page_number
                    You should answer user based on the following context and navigate the user
                    to open the right page number to know more information.

                    Context:
                    {context}
    """

    client = OpenAI(api_key="AIzaSyC0Uy7K7SveHqV0vxKwtO0O2kkVVFSxkWA",
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/") # google api
    
    response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {   "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_query
        }]
        )
    
    print(response.choices[0].message.content) 
    return response.choices[0].message.content
