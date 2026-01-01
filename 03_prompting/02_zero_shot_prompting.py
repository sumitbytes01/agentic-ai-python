from openai import OpenAI
from dotenv import load_dotenv
from os import getenv

load_dotenv()
api_key = getenv("gemini_api_key")

if not api_key:
    raise RuntimeError("gemini_api_key not found in environment variables.")

client = OpenAI(api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

# Zero shot prompting. Direct instruction to the model
SYSTEM_PROMPT = """Your name is Alex.
                You are an expert in coding and answers questions only related to coding. 
                If any other question is asked to you, just say Sorry :)"""

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hello there, can you tell me a joke?"},
        #{"role": "user", "content": "Can you write a program to calculate sum of 2 numbers?"}
        ]
)

print(response.choices[0].message.content)