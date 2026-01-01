from openai import OpenAI
from dotenv import load_dotenv
from os import getenv

load_dotenv()

api_key=getenv("gemini_api_key")
if not api_key:
    raise RuntimeError("Gemini API key not found")

client = OpenAI(api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/") # google api

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{
        "role": "user",
        "content": "Hello there, how are you!"
    }]
)

print(response.choices[0].message.content)