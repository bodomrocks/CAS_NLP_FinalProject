#!/usr/bin/env python3
"""
Schritt 1: Texte aus BGE-HTML und kantonalen TXT-Dateien extrahieren.
Ausgabe: data/raw_texts.jsonl (ein Dokument pro Zeile)

Laufzeit: ca. 5-15 Min. für alle 40K+ Dateien (mit Parallelverarbeitung)
"""

import json
import re
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BGE_DIR   = BASE_DIR / "bge_texte"
KANT_DIR  = BASE_DIR / "kantonal_texte"
OUT_FILE  = Path(__file__).resolve().parent.parent / "data" / "raw_texts.jsonl"

# Muster, die typische BGE-Metadaten-Zeilen entfernen
_META_RE = re.compile(
    r"^(BGE\s+\d+|Regeste|Sachverhalt|Erwägungen|ab Seite\s+\d+|"
    r"Urteilskopf|Dispositiv)\b",
    re.IGNORECASE
)

def clean_text(raw: str) -> str:
    # HTML-Tags entfernen
    text = re.sub(r"<[^>]+>", " ", raw)
    # Mehrfache Leerzeichen/Zeilenumbrüche normalisieren
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def process_html(path: Path) -> dict | None:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = clean_text(raw)
        if len(text) < 200:
            return None
        return {"source": "bge", "file": path.name, "text": text}
    except Exception:
        return None


def process_txt(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"[ \t]+", " ", text).strip()
        if len(text) < 200:
            return None
        return {"source": "kantonal", "file": path.name, "text": text}
    except Exception:
        return None


def main(max_bge: int = 0):
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    bge_files  = sorted(BGE_DIR.glob("*.html"))
    kant_files = sorted(KANT_DIR.glob("*.txt"))

    if max_bge and max_bge < len(bge_files):
        bge_files = bge_files[:max_bge]

    tasks: list[tuple[Path, str]] = (
        [(f, "html") for f in bge_files] +
        [(f, "txt")  for f in kant_files]
    )
    print(f"Verarbeite {len(bge_files)} BGE-Dateien + {len(kant_files)} kantonale Dateien …")

    written = 0
    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        with ProcessPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(process_html if t == "html" else process_txt, p): p
                for p, t in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extraktion"):
                result = future.result()
                if result:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    written += 1

    print(f"Fertig. {written} Dokumente → {OUT_FILE}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-bge", type=int, default=0,
                    help="Maximale Anzahl BGE-Dateien (0 = alle)")
    args = ap.parse_args()
    main(max_bge=args.max_bge)
