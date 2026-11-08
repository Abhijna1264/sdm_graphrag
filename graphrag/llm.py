"""Optional local LLM access via Ollama; every caller has an offline fallback."""
from __future__ import annotations

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"


def try_ollama(prompt: str, model: str = DEFAULT_MODEL, timeout: float = 2.0) -> str | None:
    """Return the model's response, or None if no Ollama server answers."""
    try:
        response = requests.post(
            OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False}, timeout=timeout
        )
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    text = response.json().get("response", "").strip()
    return text or None
