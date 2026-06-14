#!/usr/bin/env python3
"""
Schritt 8: Manuelle Gold-Labels in BIO-Trainingsformat umwandeln.

Teilt die 374 FVG=1-Sätze 80/20 in Train/Test auf (auf Satzebene, damit
kein Satz in beiden Splits vorkommt). Schreibt eine `gold_split`-Spalte
in eval_sentences.csv (train / test / leer für FVG=0-Zeilen).

Ausgabe:
  data/gold_train.jsonl  – 80 % der Gold-Sätze für das Training
  data/gold_test.jsonl   – 20 % der Gold-Sätze (nur Referenz, nicht im Training)
  eval_sentences.csv     – ergänzt um Spalte `gold_split`
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

EVAL_CSV   = Path(__file__).resolve().parent.parent / "data" / "eval_sentences.csv"
TRAIN_OUT  = Path(__file__).resolve().parent.parent / "data" / "gold_train.jsonl"
TEST_OUT   = Path(__file__).resolve().parent.parent / "data" / "gold_test.jsonl"

TRAIN_RATIO = 0.80
RANDOM_SEED = 42


def find_best_position(tokens: list[str], target: str,
                       near: int | None = None) -> int | None:
    positions = [i for i, t in enumerate(tokens) if t == target]
    if not positions:
        return None
    if near is None:
        return positions[0]
    return min(positions, key=lambda p: abs(p - near))


def apply_bio(labels: list[str], tokens: list[str],
              verb_tok: str, noun_tok: str, praep: str) -> bool:
    verb_pos = find_best_position(tokens, verb_tok)
    if verb_pos is None:
        return False
    noun_pos = find_best_position(tokens, noun_tok, near=verb_pos)
    if noun_pos is None:
        return False

    if praep:
        if noun_pos > 0 and tokens[noun_pos - 1].lower() == praep.lower():
            praep_pos = noun_pos - 1
        else:
            praep_pos = find_best_position(tokens, praep.capitalize())
            if praep_pos is None:
                praep_pos = find_best_position(tokens, praep)

        if praep_pos is not None and labels[praep_pos] == "O":
            labels[praep_pos] = "B-NOM"
        if labels[noun_pos] == "O":
            labels[noun_pos] = "I-NOM" if praep_pos is not None else "B-NOM"
    else:
        if labels[noun_pos] == "O":
            labels[noun_pos] = "B-NOM"

    if labels[verb_pos] == "O":
        labels[verb_pos] = "B-VERB"
    return True


def sentences_to_records(sent_groups: dict[str, list[dict]]) -> dict[str, dict]:
    """Konvertiert Satzgruppen in BIO-Records. Gibt {sentence: record} zurück."""
    records = {}
    for sentence, group in sent_groups.items():
        tokens = sentence.split(" ")
        labels = ["O"] * len(tokens)
        success = False
        for r in group:
            ok = apply_bio(labels, tokens,
                           r.get("verb_token", "").strip(),
                           r.get("noun_token", "").strip(),
                           r.get("praep", "").strip())
            if ok:
                success = True
        if success:
            records[sentence] = {"tokens": tokens, "labels": labels}
    return records


def main():
    random.seed(RANDOM_SEED)

    # CSV laden
    with open(EVAL_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    # gold_split-Spalte ergänzen falls fehlend
    if "gold_split" not in fieldnames:
        fieldnames.append("gold_split")
        for r in rows:
            r["gold_split"] = ""

    # FVG=1-Sätze gruppieren
    pos_rows = [r for r in rows if r.get("manuell_FVG", "").strip() == "1"]
    sent_groups: dict[str, list[dict]] = defaultdict(list)
    for r in pos_rows:
        sent_groups[r["sentence"]].append(r)

    unique_sents = sorted(sent_groups.keys())   # deterministisch
    random.shuffle(unique_sents)
    n_train = round(len(unique_sents) * TRAIN_RATIO)

    train_sents = set(unique_sents[:n_train])
    test_sents  = set(unique_sents[n_train:])

    print(f"Gold-Sätze gesamt : {len(unique_sents)}")
    print(f"  → Train-Split   : {len(train_sents)} Sätze")
    print(f"  → Test-Split    : {len(test_sents)} Sätze")

    # gold_split in allen Zeilen setzen
    for r in rows:
        sent = r.get("sentence", "")
        if sent in train_sents:
            r["gold_split"] = "train"
        elif sent in test_sents:
            r["gold_split"] = "test"
        # FVG=0-Zeilen bleiben leer

    # CSV zurückschreiben
    with open(EVAL_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"eval_sentences.csv aktualisiert (Spalte `gold_split`)")

    # BIO-Records erzeugen
    train_groups = {s: sent_groups[s] for s in train_sents}
    test_groups  = {s: sent_groups[s] for s in test_sents}

    train_records = sentences_to_records(train_groups)
    test_records  = sentences_to_records(test_groups)

    def write_jsonl(path, records):
        with open(path, "w", encoding="utf-8") as f:
            for rec in records.values():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_jsonl(TRAIN_OUT, train_records)
    write_jsonl(TEST_OUT,  test_records)

    print(f"✓ {len(train_records)} Sätze → {TRAIN_OUT.name}")
    print(f"✓ {len(test_records)} Sätze → {TEST_OUT.name}  (nur Referenz, nicht im Training)")


if __name__ == "__main__":
    main()
