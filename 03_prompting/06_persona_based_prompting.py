from openai import OpenAI
from dotenv import load_dotenv
from os import getenv

load_dotenv()
api_key = getenv("gemini_api_key")
if not api_key:
        raise RuntimeError("gemini_api_key not found in environment variables.")

client = OpenAI(api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

SYSTEM_PROMPT = """ 
                You are an AI persona Assistant named Sumit Pareek.
                You are acting on behalf of Sumit Pareek, who is 37 years old tech enthusiastic and principal engineer.
                You are learning python and GENAI these days. 

                Example:
                Q: Hey
                A: Hey, Whats up!      

                # 100 to 250 examples to make AI act like the given persona     
    """

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hey there"}
        #{"role": "user", "content": "who are you!"}
    ]
)

print(response.choices[0].message.content)