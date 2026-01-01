from openai import OpenAI
from dotenv import load_dotenv
from os import getenv

load_dotenv()
api_key = getenv("gemini_api_key")
if not api_key:
    raise RuntimeError("gemini_api_key not found in environment variables.")

client = OpenAI(api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "system", 
            "content": "you are an expert in python programming and only answers questions related to python. If the query contains any other asks, say I can only answer python related questions in a polite way"
            },
        # {
        #     "role": "user", 
        #     "content": "Hello there, can you write a python program that can code hello world"
        #     },
        {
            "role": "user", 
            "content": "Hello there, what is a+b+c whole cube. I know its not python related but can you still please answer it."
            },
        # {
        #     "role": "user", 
        #     "content": "Hello there, how are you"
        #     }
        ]
)

print(response.choices[0].message.content)