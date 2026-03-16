import requests
import os
from dotenv import load_dotenv

load_dotenv("d:\\Pjts\\CntextAware\\.env")
hf_token = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {hf_token}",
    "Content-Type": "application/json"
}

payload = {
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "messages": [
        {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 10
}

# The only official OpenAI compatibility endpoint for Serverless Inference
url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions"

print("Testing URL:", url)
try:
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        print(f"Success! Response: {resp.json()}")
    else:
        print(f"Error: {resp.text[:500]}")
except Exception as e:
    print(f"Failed to connect: {e}")
