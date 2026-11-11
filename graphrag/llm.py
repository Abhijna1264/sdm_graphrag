"""Gemini client tuned for the free API tier.

- Reads the key from the GEMINI_API_KEY environment variable (never from code).
- Spaces requests to stay under a requests-per-minute limit (GEMINI_RPM).
- Retries rate-limit (429) and server errors with backoff, honouring the
  retry delay Gemini returns.
- Caches every response on disk (.cache/gemini/), so rerunning a build after
  hitting the daily quota resumes instead of paying for the same calls again.

`get_llm()` returns None when no key is set; every caller then falls back to
the offline extractive path, which is what the tests use.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_EMBED_MODEL = "gemini-embedding-001"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "gemini"


class QuotaExhausted(RuntimeError):
    """Raised when the daily free-tier quota is used up; rerun later to resume."""


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        rpm: float = 10.0,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.embed_model = embed_model
        self.min_interval = 60.0 / rpm
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_call = 0.0
        self.calls = 0  # real API requests made (cache hits excluded)
        self.cache_hits = 0

    # -- plumbing ---------------------------------------------------------

    def _cache_path(self, kind: str, payload: dict) -> Path:
        digest = hashlib.sha256(json.dumps([kind, payload], sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _post(self, url: str, body: dict) -> dict:
        for attempt in range(8):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                response = requests.post(
                    url, json=body, headers={"x-goog-api-key": self.api_key}, timeout=120
                )
            except requests.RequestException:
                time.sleep(5 * 2 ** attempt)
                continue
            self.calls += 1
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:
                text = response.text
                if "PerDay" in text or "per day" in text.lower():
                    raise QuotaExhausted(
                        "Gemini daily free-tier quota used up. Responses so far are cached; "
                        "rerun the same command tomorrow (or with another model) to resume."
                    )
                match = re.search(r'"retryDelay":\s*"(\d+)', text)
                time.sleep(int(match.group(1)) + 1 if match else 10 * 2 ** attempt)
                continue
            if response.status_code >= 500:
                time.sleep(5 * 2 ** attempt)
                continue
            raise RuntimeError(f"Gemini API error {response.status_code}: {response.text[:500]}")
        raise RuntimeError("Gemini API kept failing after retries.")

    # -- public API -------------------------------------------------------

    def generate(self, prompt: str, json_output: bool = False, temperature: float = 0.0) -> str:
        config = {"temperature": temperature}
        if json_output:
            config["responseMimeType"] = "application/json"
        payload = {"model": self.model, "prompt": prompt, "config": config}
        path = self._cache_path("generate", payload)
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))

        data = self._post(
            f"{API_ROOT}/{self.model}:generateContent",
            {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config},
        )
        try:
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError):
            text = ""  # blocked or empty response; don't retry forever
        path.write_text(json.dumps(text), encoding="utf-8")
        return text

    def generate_json(self, prompt: str) -> dict | list | None:
        text = self.generate(prompt, json_output=True)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"[\[{].*[\]}]", text, re.S)
            try:
                return json.loads(match.group()) if match else None
            except json.JSONDecodeError:
                return None

    def embed(self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        vectors: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for i, text in enumerate(texts):
            path = self._cache_path("embed", {"model": self.embed_model, "task": task, "text": text})
            if path.exists():
                self.cache_hits += 1
                vectors[i] = json.loads(path.read_text(encoding="utf-8"))
            else:
                missing.append(i)

        for start in range(0, len(missing), 100):  # batchEmbedContents takes up to 100
            batch = missing[start : start + 100]
            data = self._post(
                f"{API_ROOT}/{self.embed_model}:batchEmbedContents",
                {
                    "requests": [
                        {
                            "model": f"models/{self.embed_model}",
                            "content": {"parts": [{"text": texts[i]}]},
                            "taskType": task,
                        }
                        for i in batch
                    ]
                },
            )
            for i, item in zip(batch, data["embeddings"]):
                vectors[i] = item["values"]
                path = self._cache_path("embed", {"model": self.embed_model, "task": task, "text": texts[i]})
                path.write_text(json.dumps(item["values"]), encoding="utf-8")
        return vectors  # type: ignore[return-value]


def _windows_user_env(name: str) -> str | None:
    """Read a variable saved with `setx`. Terminals opened inside an app that was
    already running (VS Code, etc.) don't inherit it until that app restarts."""
    if os.name != "nt":
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return None
    return str(value).strip() or None


def get_llm() -> GeminiClient | None:
    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or _windows_user_env("GEMINI_API_KEY")
    )
    if not api_key:
        return None
    return GeminiClient(
        api_key,
        model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
        embed_model=os.environ.get("GEMINI_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        rpm=float(os.environ.get("GEMINI_RPM", "10")),
    )
