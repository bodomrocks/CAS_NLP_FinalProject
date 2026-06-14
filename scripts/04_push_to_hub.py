#!/usr/bin/env python3
"""
Schritt 4: Trainiertes Modell auf den Hugging Face Hub pushen.

Voraussetzungen:
  - Entweder vorher: huggingface-cli login
  - Oder: --token hf_xxxx  (Write Token direkt angeben)

Verwendung:
  python 04_push_to_hub.py --repo IhrUsername/fvg-gbert-base
  python 04_push_to_hub.py --repo IhrUsername/fvg-gbert-base --token hf_xxxx
"""

import argparse
from pathlib import Path
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "fvg-gbert-base"


def main(repo_id: str, token: str | None):
    if token:
        login(token=token)
        print("Token-Login erfolgreich.")

    print(f"Lade Modell aus: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model     = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR))

    print(f"Pushe auf HF Hub: {repo_id} …")
    tokenizer.push_to_hub(repo_id)
    model.push_to_hub(repo_id)

    print(f"\nFertig! Modell verfügbar unter:")
    print(f"  https://huggingface.co/{repo_id}")
    print(f"\nNächster Schritt: MODEL_ID in app/app.py auf '{repo_id}' setzen.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="HF Hub Repo-ID, z.B. meinname/fvg-gbert-base")
    ap.add_argument("--token", default=None,
                    help="HF Write Token (alternativ zu huggingface-cli login)")
    args = ap.parse_args()
    main(args.repo, args.token)
