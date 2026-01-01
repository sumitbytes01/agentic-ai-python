from openai import OpenAI
from dotenv import load_dotenv
from os import getenv

load_dotenv()
api_key = getenv("gemini_api_key")
if not api_key:
    raise RuntimeError("gemini_api_key not found in environment variables.")

client = OpenAI(api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

# few shot prompting. Direct instruction along with few examples to the model
SYSTEM_PROMPT = """Your name is Alex.
                    You are an expert in python programming and only 
                    answers python related questions. 
                    If the query contains other asks, 
                    say I can only answer python related questions.
                    
                Examples:
                Q:Can you tell me the capital of india ?
                A: Sorry, I can answer only Python related questions and my name is Alex.

                Q: Please write a Python program to add two numbers.
                A: def add(a,b):
                        return a+b
                    """

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        #{"role": "user", "content": "Hello there, can you tell me a joke?"}
        {"role": "user", "content": "Hello there, can you write a program to multiply 2 numbers"}
        ]
)

print(response.choices[0].message.content)