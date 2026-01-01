# source: https://huggingface.co/google/gemma-3-4b-it
# Hugging Face - Github of LLM models

from transformers import pipeline

pipe = pipeline("image-text-to-text", model="google/gemma-3-4b-it")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image", 
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {
                "type": "text", 
                "text": "What animal is on the candy?"}
        ]
     }
]
output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])