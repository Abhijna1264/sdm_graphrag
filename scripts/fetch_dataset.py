#!/usr/bin/env python3
"""CLI: download the evaluation corpus (Wikipedia articles on space exploration).

Each article's plain-text body is truncated to --max-chars so the whole
corpus stays small enough to index on the Gemini free tier (roughly 60k
tokens -> ~50 extraction calls). Wikipedia text is CC BY-SA 4.0.

Usage:
    python scripts/fetch_dataset.py --out data/corpus
"""
import argparse
import re
import time
from pathlib import Path

import requests

API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "sdm-graphrag-course-project/1.0 (educational use)"}

ARTICLES = [
    "Space Race", "Sputnik 1", "Vostok 1", "Yuri Gagarin", "Project Mercury",
    "Project Gemini", "Apollo program", "Apollo 11", "Neil Armstrong", "Apollo 13",
    "Wernher von Braun", "Sergei Korolev", "NASA", "Soviet space program",
    "Saturn V", "Space Shuttle program", "Mir", "International Space Station",
    "European Space Agency", "Indian Space Research Organisation", "Chandrayaan-3",
    "China National Space Administration", "Chang'e 4", "Hubble Space Telescope",
    "James Webb Space Telescope", "Voyager program", "Mars rover", "Perseverance (rover)",
    "SpaceX", "Falcon 9", "Artemis program", "Space debris",
]


def fetch_extract(title: str) -> str:
    params = {
        "action": "query", "format": "json", "prop": "extracts", "explaintext": 1,
        "redirects": 1, "titles": title,
    }
    for attempt in range(6):
        response = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
        if response.status_code != 429:
            break
        time.sleep(int(response.headers.get("Retry-After", 0)) or 5 * 2 ** attempt)
    response.raise_for_status()
    pages = response.json()["query"]["pages"]
    return next(iter(pages.values())).get("extract", "")


def clean(text: str, max_chars: int) -> str:
    # Drop trailing reference-style sections and section headings.
    text = re.split(r"\n==\s*(See also|References|Notes|Further reading|External links)\s*==", text)[0]
    text = re.sub(r"\n=+[^=\n]+=+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    return cut[: cut.rfind(".") + 1] or cut


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Wikipedia space-exploration corpus.")
    parser.add_argument("--out", default="data/corpus", help="Output directory for .txt files")
    parser.add_argument("--max-chars", type=int, default=8000, help="Max characters kept per article")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for title in ARTICLES:
        name = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
        path = out / f"{name}.txt"
        if path.exists():  # re-runs resume instead of re-downloading
            total += len(path.read_text(encoding="utf-8"))
            continue
        text = clean(fetch_extract(title), args.max_chars)
        if not text:
            print(f"  ! no text for {title}")
            continue
        path.write_text(f"{title}\n{text}\n", encoding="utf-8")
        total += len(text)
        print(f"  {title}: {len(text)} chars")
        time.sleep(2)
    print(f"Saved {len(ARTICLES)} articles, {total} chars (~{total // 4} tokens) to {out}")


if __name__ == "__main__":
    main()
