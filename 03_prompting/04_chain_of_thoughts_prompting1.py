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

# chain of thoughts prompting. Direct instruction to the model to think in steps.
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

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    response_format={"type": "json_object"},  # this forces the model to respond in JSON format
    messages=[
        {
                "role": "system", 
                "content": SYSTEM_PROMPT},
        {
                "role": "user",
                "content": "Hello there, write a program to add n numbers using python",
        },
        # manually keep on adding messages to create History
        {
                "role": "assistant",
                "content": json.dumps(
                        {
                        "Step": "START",
                        "content": "Hello there, write a program to add n numbers using python",
                        }
            ),
        },
        # {
        #     "role": "assistant",
        #     "content": json.dumps(
        #         {
        #             "Step": "PLAN",
        #             "content": "The user wants a Python program to add 'n' numbers. I will provide a clear, concise program for this task.",
        #         }
        #     ),
        # },
        # {
        #     "role": "assistant",
        #     "content": json.dumps(
        #         {
        #             "Step": "PLAN",
        #             "content": "First, I'll consider how to get 'n' numbers from the user. A common approach is to take 'n' as an input first, and then take 'n' individual numbers, or to take a comma-separated string of numbers.",
        #         }
        #     ),
        # },
        # {
        #     "role": "assistant",
        #     "content": json.dumps(
        #         {
        #             "Step": "PLAN",
        #             "content": "I will opt for taking a space-separated string of numbers from the user, which is a flexible way to handle 'n' numbers.",
        #         }
        #     ),
        # },
        # {
        #     "role": "assistant",
        #     "content": json.dumps(
        #         {
        #             "Step": "PLAN",
        #             "content": "The plan is to: 1. Prompt the user to enter numbers, separated by spaces. 2. Read the input string. 3. Split the string into individual number strings. 4. Convert each number string to an integer (or float). 5. Use the built-in `sum()` function to calculate the total. 6. Print the result.",
        #         }
        #     ),
        # },
    ],
)

print(response.choices[0].message.content)
