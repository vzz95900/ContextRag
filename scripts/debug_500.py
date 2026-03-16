"""Test chat API endpoint directly to get the traceback."""
import sys
import traceback
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(r"d:\Pjts\CntextAware")
sys.path.insert(0, str(ROOT))

try:
    from app.main import app
    client = TestClient(app)
    
    print("Sending request to /api/chat...")
    response = client.post(
        "/api/chat",
        json={"query": "What is this document about?"}
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code != 200:
        print("Error detail:")
        print(response.json())
        sys.exit(1)
        
    print("Success!")
    print(response.json()["answer"][:100])
except Exception as e:
    print(f"ERROR:\n{traceback.format_exc()}")
    sys.exit(1)
