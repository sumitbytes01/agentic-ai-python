# pip install mem0ai
from dotenv import load_dotenv
from mem0 import Memory
import os
import json

from google import genai
from os import getenv

load_dotenv()

api_key=getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError("Gemini API key not found")

client = genai.Client(
    api_key=api_key
)

config = {
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/text-embedding-004",
        }
    },
        "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-2.5-flash",
            "temperature": 0.0,
            "max_tokens": 1000,
        }
    },
    "vector_store": {
        "config": {
            "embedding_model_dims": 768,
        }
    }
}

mem_client = Memory.from_config(config)


while True:

    user_query = input("> ")

    search_memory = mem_client.search(query=user_query, user_id="sumit pareek",)

    memories = [
        f"ID: {mem.get("id")}\nMemory: {mem.get("memory")}" 
        for mem in search_memory.get("results")
    ]

    print("Found Memories", memories)

    SYSTEM_PROMPT = f"""
        Here is the context about the user:
        {json.dumps(memories)}
    """

    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=user_query,
    config={
        "system_instruction": SYSTEM_PROMPT
    }
)

    ai_response = response.text

    print("AI:", ai_response)

    mem_client.add(
        user_id="sumit pareek",
        messages=[
            { "role": "user", "content": user_query },
            { "role": "assistant", "content": ai_response }
        ]
    )

    print("Memory has been saved...")