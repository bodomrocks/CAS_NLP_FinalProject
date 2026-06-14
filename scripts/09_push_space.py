#!/usr/bin/env python3
"""
Schritt 9: App-Dateien auf HuggingFace Spaces pushen.

Lädt app.py, fvg_syntax_filter.py und requirements.txt
in das angegebene Space-Repository hoch.

Verwendung:
  python 09_push_space.py --space boj-per/legal_fvg2
  python 09_push_space.py --space boj-per/legal_fvg2 --token hf_xxxx
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, login

APP_DIR = Path(__file__).resolve().parent.parent / "app"

FILES = [
    "app.py",
    "fvg_syntax_filter.py",
    "requirements.txt",
]


def main(space_id: str, token: str | None):
    if token:
        login(token=token)

    api = HfApi()

    print(f"Pushe App-Dateien → {space_id} …")
    for filename in FILES:
        path = APP_DIR / filename
        if not path.exists():
            print(f"  ⚠ Nicht gefunden: {filename} – übersprungen")
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=filename,
            repo_id=space_id,
            repo_type="space",
        )
        print(f"  ✓ {filename}")

    print(f"\nFertig! Space: https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="boj-per/legal_fvg2")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()
    main(args.space, args.token)
