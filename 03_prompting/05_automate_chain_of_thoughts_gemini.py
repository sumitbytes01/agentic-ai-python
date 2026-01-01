from openai import OpenAI
from dotenv import load_dotenv
from os import getenv
import json

load_dotenv()
api_key = getenv("gemini_api_key")

if not api_key:
        raise RuntimeError("gemini_api_key not found in environment variables.")

client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


# few shot prompting. Direct instruction along with few examples to the model
SYSTEM_PROMPT = """
                You are an expert AI assistant in resolving user queries using chain of thoughts.
                You work on START, PLAN and OUTPUT steps.
                You need to first PLAN what needs to be done. PLAN can have multiple steps.
                Once you think enough PLAN has been done, finally you can give an OUTPUT.

                Rules:
                - stricly follow the OUTPUT in JSON format.
                - Only run one step at a time.
                - The sequence of steps is START(where user gives an input), 
                                           PLAN(That can be multiple times) and finally
                                           OUTPUT(Which is going to be displayed to the user.)

                Output JSON format:
                {
                        "Step": "START" | "PLAN" | "OUTPUT", 
                        "content": "string"
                }

                Example: 
                Plan: {"Step":"START",
                        "content" : "Hey, can you solve 2+3*5/10"}
                Plan: {"Step":"PLAN",
                        "content" : "Seems like user is interested in math problem"}
                Plan: {"Step":"PLAN",
                        "content" : "Looking at the problem we should solve it using BODMAS method"}
                Plan: {"Step":"PLAN",
                        "content" : "Yes, the BODMAS is the correct thing to do here"}
                Plan: {"Step":"PLAN",
                        "content" : "first we divide 5/10 and that gives us 0.5, making the equation as: 2+3*0.5"}
                Plan: {"Step":"PLAN",
                        "content" : "then we perform the multiplication operation between 3 and 0.5 making the equation as: 2+1.5"}
                Plan: {"Step":"PLAN",
                        "content" : "finally, we perform the add operation between 2 and 1.5 making the equation as: 3.5"}
                Plan: {"Step":"OUTPUT",
                        "content" : "Finally, we are able to solve the equation and the OUTPUT is 3.5"}
                    """

message_history = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

user_query = input("✍️->")
message_history.append({"role":"user", "content": user_query})

# convert to equivalent genai code
while True:
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        response_format={"type": "json_object"}, # this forces the model to respond in JSON format
        messages=message_history
    )

    raw_result = response.choices[0].message.content
    message_history.append({"role":"assistant", "content": raw_result})

    parsed_result = json.loads(raw_result)

    if parsed_result.get("Step") == "START":
        print("Thinking.🧠..", parsed_result.get("content"))
        continue

    if parsed_result.get("Step") == "PLAN":
        print("Working on it.🧑‍💻..", parsed_result.get("content"))
        continue

    if parsed_result.get("Step") == "OUTPUT":
        print("Eureka...", parsed_result.get("content"))
        break