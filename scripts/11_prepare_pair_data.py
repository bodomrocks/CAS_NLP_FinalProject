#!/usr/bin/env python3
"""
Schritt 11: Trainingsdaten für den Paarklassifikator vorbereiten.

Liest eval_sentences.csv und erzeugt:
  - data/pair_train.jsonl  (Positive: gold_split=train; Negative: 80% der FVG=0-Zeilen)
  - data/pair_test.jsonl   (Positive: gold_split=test;  Negative: 20% der FVG=0-Zeilen)

Eingabeformat für den Klassifikator:
  text_a : Satz
  text_b : "{verb_token} {pattern}"   (pattern = praep noun_token | noun_token)
  label  : 1 (FVG) | 0 (kein FVG)
"""

import csv
import json
import random
from pathlib import Path

EVAL_CSV   = Path(__file__).resolve().parent.parent / "data" / "eval_sentences.csv"
TRAIN_OUT  = Path(__file__).resolve().parent.parent / "data" / "pair_train.jsonl"
TEST_OUT   = Path(__file__).resolve().parent.parent / "data" / "pair_test.jsonl"

SEED = 42


def make_record(row: dict, label: int) -> dict:
    praep    = row.get("praep", "").strip()
    verb_tok = row.get("verb_token", "").strip()
    noun_tok = row.get("noun_token", "").strip()
    pattern  = f"{praep} {noun_tok}".strip() if praep else noun_tok
    return {
        "sentence"  : row["sentence"].strip(),
        "verb_token": verb_tok,
        "noun_token": noun_tok,
        "praep"     : praep,
        "text_b"    : f"{verb_tok} {pattern}",
        "label"     : label,
        # Metadaten für spätere Fehleranalyse
        "verb_lemma": row.get("verb_lemma", ""),
        "noun_lemma": row.get("noun_lemma", ""),
        "gold_split": row.get("gold_split", ""),
    }


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  {path.name}: {len(records)} Sätze "
          f"({sum(r['label'] for r in records)} pos / "
          f"{sum(1-r['label'] for r in records)} neg)")


def main():
    with open(EVAL_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    pos_train = [r for r in rows if r.get("gold_split", "").strip() == "train"]
    pos_test  = [r for r in rows if r.get("gold_split", "").strip() == "test"]
    negatives = [r for r in rows if r["manuell_FVG"] == "0"]

    random.seed(SEED)
    random.shuffle(negatives)
    split_idx  = int(len(negatives) * 0.8)
    neg_train  = negatives[:split_idx]
    neg_test   = negatives[split_idx:]

    train_records = (
        [make_record(r, 1) for r in pos_train] +
        [make_record(r, 0) for r in neg_train]
    )
    test_records = (
        [make_record(r, 1) for r in pos_test] +
        [make_record(r, 0) for r in neg_test]
    )

    random.shuffle(train_records)

    print("Schreibe Paar-Datensätze …")
    write_jsonl(train_records, TRAIN_OUT)
    write_jsonl(test_records,  TEST_OUT)
    print("Fertig.")


if __name__ == "__main__":
    main()
