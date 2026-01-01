# fastapi dev ollama_connect.py
# pip install ollama , to get the ollama sdk.
from fastapi import FastAPI, Body
from ollama import chat

app = FastAPI()

# Use an HTTP method decorator (post/get/put/...) — Body suggests POST
@app.post("/color-check")
def color_check(message: str = Body(..., description="why is the sky blue")):
    # Send the user's message to the ollama `chat` call
    response = chat(
            model='llama3.1', 
            messages=[
                {
                    "role": 'user', 
                    'content': message
                }])
    # Be defensive about the response shape since SDKs vary.
    # Prefer the common `response.message.content` if available, otherwise fall back to str()
    try:
        content = response.message.content
    except Exception:
        try:
            # sometimes the SDK returns a list or different shape
            content = response[0].message.content
        except Exception:
            content = str(response)

    return {"response": content}