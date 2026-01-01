from openai import OpenAI
from dotenv import load_dotenv
from os import getenv

load_dotenv()
api_key = getenv("gemini_api_key")
if not api_key:
    raise RuntimeError("gemini_api_key not found in environment variables.")

client = OpenAI(api_key=api_key ,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

# few shot prompting. Direct instruction along with few examples to the model
SYSTEM_PROMPT = """ Your name is Alex.
                    You are an expert in python programming and only 
                    answer python related questions. If the query contains other asks, 
                    say I can only answer python related questions.

                Rule:
                - stricly follow the output in JSON format.

                Output Format:
                {{
                    "code": "string" | None,
                    "isCodingQuestion": boolean
                }}

                Examples:
                Q:Can you tell me the capital of india ?
                A: {{
                    "code": None,
                    "isCodingQuestion": alse
                    }}

                Q: Please write a Python program to add two numbers.
                A: {{
                    "code":"def add(a,b):
                                return a+b",
                    "isCodingQuestion": true
                    }}
                    """

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    response_format={"type": "json_object"}, # this forces the model to respond in JSON format
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        #{"role": "user", "content": "Hello there, can you tell me a joke?"}
        {"role": "user", "content": "Hello there, can you write a python program to print numbers 1 to 10 in for loop"}
        ]
)
content = response.choices[0].message.content
print(content)