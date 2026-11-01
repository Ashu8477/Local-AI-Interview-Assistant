"""Local LLM service using Ollama HTTP API with streaming."""

import json
import requests

from app.utils.config import OLLAMA_BASE_URL, OLLAMA_MODEL, SYSTEM_PROMPT


class LLMService:
    """Communicates with a locally running Ollama instance."""

    def __init__(self):
        self.base_url = OLLAMA_BASE_URL.rstrip("/")

    def check_connection(self) -> tuple[bool, str]:
        """Check if Ollama is reachable. Returns (ok, message)."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                return True, "Ollama is running."
            return False, f"Ollama returned status {resp.status_code}."
        except requests.ConnectionError:
            return False, "Cannot connect to Ollama. Is it installed and running?"
        except Exception as e:
            return False, f"Ollama check failed: {e}"

    def check_model(self) -> tuple[bool, str]:
        """Check if the configured model is available. Returns (ok, message)."""
        ok, msg = self.check_connection()
        if not ok:
            return False, msg
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            if any(OLLAMA_MODEL in m for m in models):
                return True, f"Model '{OLLAMA_MODEL}' is available."
            return False, (
                f"Model '{OLLAMA_MODEL}' not found. "
                f"Run: ollama pull {OLLAMA_MODEL}"
            )
        except Exception as e:
            return False, f"Failed to check models: {e}"

    def generate_stream(self, prompt: str):
        """Generator that yields text tokens from Ollama streaming API.

        Yields (token_str, error_str) tuples. On success error_str is empty.
        On final done, token_str is empty and error_str is empty.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": True,
        }
        try:
            with requests.post(url, json=payload, stream=True, timeout=120) as resp:
                if resp.status_code != 200:
                    yield "", f"Ollama returned status {resp.status_code}: {resp.text[:200]}"
                    return
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("error"):
                        yield "", f"Ollama error: {data['error']}"
                        return
                    token = data.get("response", "")
                    done = data.get("done", False)
                    if token:
                        yield token, ""
                    if done:
                        yield "", ""
                        return
        except requests.ConnectionError:
            yield "", "Lost connection to Ollama during generation."
        except Exception as e:
            yield "", f"Generation failed: {e}"
