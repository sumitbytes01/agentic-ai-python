from dotenv import load_dotenv
from openai import OpenAI
from os import getenv

load_dotenv()
api_key=getenv("GEMINI_API_KEY")    
if not api_key:
    raise RuntimeError("Gemini API key not found")

client = OpenAI(api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/") # google api


response = client.chat.completions.create(
    model="gemini-2.5-flash", 
    messages=[
        {
            "role": "user",
            "content": [
                { "type": "text", "text": "Generate a caption for this image in about 50 words" },
                { "type": "image_url", 
                    "image_url": {
                        "url": "https://images.pexels.com/photos/879109/pexels-photo-879109.jpeg"
                        } }
            ]
         }
    ]
)

print("Response:", response.choices[0].message.content)