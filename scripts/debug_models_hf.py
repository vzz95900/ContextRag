import requests
import os
from dotenv import load_dotenv

load_dotenv("d:\\Pjts\\CntextAware\\.env")
hf_token = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {hf_token}",
    "Content-Type": "application/json"
}

models_to_test = [
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "HuggingFaceH4/zephyr-7b-beta",
    "microsoft/Phi-3.5-mini-instruct"
]

url = "https://router.huggingface.co/hf-inference/v1/chat/completions"

for model in models_to_test:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10
    }
    resp = requests.post(url, headers=headers, json=payload)
    print(f"Model: {model}")
    print(f"  Status: {resp.status_code}")
    if resp.status_code == 200:
        print("  ✅ WORKS!")
        break
    else:
        print(f"  ❌ Failed: {resp.text[:100]}")
    print()
