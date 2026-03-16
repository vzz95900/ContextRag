import requests
import os
from dotenv import load_dotenv

load_dotenv("d:\\Pjts\\CntextAware\\.env")
hf_token = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {hf_token}",
    "Content-Type": "application/json"
}

model = "mistralai/Mistral-7B-Instruct-v0.3"

urls = [
    f"https://router.huggingface.co/{model}/v1/chat/completions",
    f"https://router.huggingface.co/models/{model}/v1/chat/completions",
    f"https://router.huggingface.co/v1/chat/completions",
    f"https://api-inference.huggingface.co/models/{model}"  # Legacy generic API
]

for url in urls:
    if "v1/chat/completions" in url:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10
        }
    else:
        payload = {
            "inputs": "<s>[INST] Hi [/INST]",
            "parameters": {"max_new_tokens": 10}
        }
        
    print(f"Testing URL: {url}")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            print("  ✅ WORKS!")
        else:
            print(f"  ❌ Failed: {resp.text[:100]}")
    except Exception as e:
        print(f"  Network error: {e}")
    print()
