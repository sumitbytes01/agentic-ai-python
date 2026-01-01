# pip install google-genai

from google import genai
from os import getenv
from dotenv import load_dotenv

load_dotenv()

api_key=getenv("gemini_api_key")
if not api_key:
    raise RuntimeError("Gemini API key not found")

client = genai.Client(
    api_key=api_key
)

response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents="Hello, how are you!"
)

print("Response is: ", response.text)