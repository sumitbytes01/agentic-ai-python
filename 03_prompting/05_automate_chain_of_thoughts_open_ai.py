from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

client = OpenAI()


# few shot prompting. Direct instruction along with few examples to the model
SYSTEM_PROMPT = """
                        You are an expert AI assistant in resolving user queries using chain of thoughts.
                        You work on START, PLAN and OUTPUT steps.
                        You need to first PLAN what needs to be done. PLAN can have multiple steps.
                        Once you think enough PLAN has been done, finally you can give an OUTPUT.
                Rules:
                - stricly follow the OUTPUT in JSON format.
                - Only run one step at a time.
                - The sequence of steps is START(where user gives an input), PLAN(That can be multiple times) and finally
                OUTPUT(Which is going to be displayed to the user.)

                OutputJSON format:
                {
                        "Step": "START" | "PLAN" | "OUTPUT", 
                        "content": "string"
                }

                Example: 
                Plan: {"step":"START",
                        "content" : "Hey, can you solve 2+3*5/10"}
                Plan: {"step":"PLAN",
                        "content" : "Seems like user is interested in math problem"}
                Plan: {"step":"PLAN",
                        "content" : "Looking at the problem we should solve it using BODMAS method"}
                Plan: {"step":"PLAN",
                        "content" : "Yes, the BODMAS is the correct thing to do here"}
                Plan: {"step":"PLAN",
                        "content" : "first we divide 5/10 and that gives us 0.5, making the equation as: 2+3*0.5"}
                Plan: {"step":"PLAN",
                        "content" : "then we perform the multiplication operation between 3 and 0.5 making the equation as: 2+1.5"}
                Plan: {"step":"PLAN",
                        "content" : "finally, we perform the add operation between 2 and 1.5 making the equation as: 3.5"}
                Plan: {"step":"OUTPUT",
                        "content" : "Finally, we are able to solve the equation and the OUTPUT is 3.5"}
                    """

message_history = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

user_query = input("->")
message_history.append({"role":"user", "content": user_query})

# convert to equivalent genai code
while True:
    response = client.chat.completions.create(
        model = "gpt-5-nano",
        response_format={"type": "json_object"},
        messages=message_history
    )

    raw_result = response.choices[0].message.content
    message_history.append({"role":"assistant", "content": raw_result})

    parsed_result = json.loads(raw_result)

    if parsed_result.get("step") == "Start":
        print("Thinking...", parsed_result.get("content"))
        continue
    
    if parsed_result.get("step") == "Plan":
        print("Working on it...", parsed_result.get("content"))
        continue

    if parsed_result.get("step") == "Output":
        print("Eureka...", parsed_result.get("content"))
        break