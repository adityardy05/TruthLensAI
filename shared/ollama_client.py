import os
import time
import json
import requests
from functools import wraps

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME      = os.getenv("OLLAMA_MODEL",    "deepseek-r1:7b")


def with_retry(max_attempts=3, delay=2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.ConnectionError as e:
                    last_error = f"Ollama unreachable: {e}"
                    if attempt < max_attempts:
                        time.sleep(delay * attempt)
                except requests.exceptions.Timeout as e:
                    last_error = f"Timed out: {e}"
                    if attempt < max_attempts:
                        time.sleep(delay)
                except Exception as e:
                    raise e
            raise RuntimeError(f"Failed after {max_attempts} attempts: {last_error}")
        return wrapper
    return decorator


@with_retry(max_attempts=3, delay=2.0)
def call_deepseek(prompt: str, timeout: int = 120) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model":   MODEL_NAME,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 512}
        },
        timeout=timeout
    )
    response.raise_for_status()
    return response.json()["response"].strip()


@with_retry(max_attempts=3, delay=2.0)
def call_deepseek_json(prompt: str, timeout: int = 120) -> dict:
    raw   = call_deepseek(prompt, timeout)
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1])
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from model: {e}\nRaw: {raw[:200]}")


def check_ollama_health() -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models   = [m["name"] for m in response.json().get("models", [])]
        return any(MODEL_NAME.split(":")[0] in m for m in models)
    except Exception as e:
        print(f"Ollama health check failed: {e}")
        return False