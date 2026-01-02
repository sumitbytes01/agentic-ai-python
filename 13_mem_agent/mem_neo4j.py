# pip install mem0ai
from dotenv import load_dotenv
from mem0 import Memory
import os
import json

from ollama import chat
from os import getenv

load_dotenv()

get_neo4j = getenv("NEO_GRAPH_STORE")
if not get_neo4j:
    raise RuntimeError("Neo4j Graph Store password not found")

config = {
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
        }
    },
        "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.1",
            "temperature": 0.0,
            "max_tokens": 1000,
        }
    },
    "vector_store": {
        "config": {
            "embedding_model_dims": 768,
        }
    },
    "graph_store":{
        "provider": "neo4j",
        "config": {
            "url": "neo4j+s://1b4d6951.databases.neo4j.io",
            "username": "neo4j",
            "password": get_neo4jhe
        }
    }
}

mem_client = Memory.from_config(config)


while True:

    user_query = input("> ")

    search_memory = mem_client.search(query=user_query, user_id="sumit pareek",)

    memories = [
        f"ID: {mem.get('id')}\nMemory: {mem.get('memory')}" 
        for mem in search_memory.get("results")
    ]

    print("Found Memories", memories)

    SYSTEM_PROMPT = f"""
        Here is the context about the user:
        {json.dumps(memories)}
    """

    response = chat(
    model='llama3.1',
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
)

    ai_response = response.message.content

    print("AI:", ai_response)

    mem_client.add(
        user_id="sumit pareek",
        messages=[
            { "role": "user", "content": user_query },
            { "role": "assistant", "content": ai_response }
        ]
    )

    print("Memory has been saved...")