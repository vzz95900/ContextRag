"""List available Gemini models that support embedding."""
import sys
from pathlib import Path

ROOT = Path(r"d:\Pjts\CntextAware")
sys.path.insert(0, str(ROOT))
LOG = ROOT / "data" / "models_debug.log"

lines = []
def log(msg):
    print(msg)
    lines.append(str(msg))

try:
    from google import genai
    from app.core.config import settings

    client = genai.Client(api_key=settings.gemini_api_key)

    # Test 1: List models that support embedding
    log("=== Models supporting embedContent ===")
    for model in client.models.list():
        methods = model.supported_actions if hasattr(model, 'supported_actions') else []
        name = model.name if hasattr(model, 'name') else str(model)
        # Check if model supports embedding
        if 'embed' in str(name).lower() or 'embedding' in str(name).lower():
            log(f"  {name}")

    # Test 2: Quick generation test to verify API key
    log("\n=== Testing API key with a simple generation ===")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Say hello in one word.",
    )
    log(f"  Generation OK: {response.text}")

    # Test 3: Try embedding with different model names
    log("\n=== Testing embedding models ===")
    test_models = [
        "text-embedding-004",
        "models/text-embedding-004",
        "embedding-001",
        "models/embedding-001",
        "gemini-embedding-exp-03-07",
    ]
    for m in test_models:
        try:
            result = client.models.embed_content(model=m, contents="test")
            dim = len(result.embeddings[0].values)
            log(f"  {m} -> OK (dim={dim})")
            break
        except Exception as e:
            err = str(e)[:80]
            log(f"  {m} -> FAIL: {err}")

except Exception as e:
    log(f"ERROR: {e}")

LOG.write_text("\n".join(lines), encoding="utf-8")
