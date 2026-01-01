from huggingface_hub import InferenceClient
from os import getenv
from dotenv import load_dotenv

load_dotenv()
hf_token = getenv("HF_TOKEN")

if not hf_token:
    raise RuntimeError("HF_TOKEN not found in environment variables.")

# Initialize client with your HF token
client = InferenceClient(
    model="Qwen/Qwen2-VL-7B-Instruct",
    token=hf_token
)

# Prepare your prompt and image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Send request to the hosted model
output = client.chat_completion(
    messages=messages,
    max_tokens=200
)

print(output.choices[0].message["content"])
