#!/usr/bin/env python3
"""
Schritt 7: Modell-Evaluation gegen manuelles Gold-Label.

Lädt boj-per/fvg-gbert-base, führt Inferenz auf allen Sätzen in
eval_sentences.csv durch und prüft, ob das spezifische Verb-Nomen-Paar
als FVG erkannt wird.

Ausgabe:
  - data/eval_model_results.csv  (mit Modellvorhersage pro Zeile)
  - Konsolenausgabe mit Precision / Recall / F1
"""

import csv
import sys
from pathlib import Path

import spacy
import torch
from transformers import pipeline

# Syntaktischen Filter aus App-Modul importieren
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from fvg_syntax_filter import syntactic_filter

MODEL_ID = "boj-per/fvg-gbert-base"
EVAL_CSV  = Path(__file__).resolve().parent.parent / "data" / "eval_sentences.csv"
OUT_CSV   = Path(__file__).resolve().parent.parent / "data" / "eval_model_results.csv"

BATCH_SIZE = 32


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader.fieldnames or []), list(reader)


def find_char_offset(sentence: str, token: str) -> list[int]:
    """Gibt alle Zeichenpositionen zurück, an denen 'token' in 'sentence' beginnt."""
    pos, offsets = 0, []
    while True:
        idx = sentence.find(token, pos)
        if idx == -1:
            break
        # Sicherstellen, dass es ein ganzes Wort ist (Leerzeichen / Satzanfang / Satzende)
        before = idx == 0 or sentence[idx - 1] == " "
        after  = idx + len(token) == len(sentence) or sentence[idx + len(token)] in " .,;:!?\"'()"
        if before and after:
            offsets.append(idx)
        pos = idx + 1
    return offsets


def model_predicts_fvg(entities, sentence, verb_tok, noun_tok, praep_tok="") -> bool:
    """
    Gibt True zurück, wenn das Modell verb_tok als VERB und noun_tok als NOM
    (im selben FVG-Kontext) markiert hat.
    """
    verb_entities = [e for e in entities if e["entity_group"] == "VERB"]
    nom_entities  = [e for e in entities if e["entity_group"] == "NOM"]

    verb_offsets = find_char_offset(sentence, verb_tok)
    noun_offsets = find_char_offset(sentence, noun_tok)

    verb_hit = any(
        any(e["start"] <= vo < e["end"] for e in verb_entities)
        for vo in verb_offsets
    )
    noun_hit = any(
        any(e["start"] <= no < e["end"] for e in nom_entities)
        for no in noun_offsets
    )

    return verb_hit and noun_hit


def compute_metrics(results: list[dict], label_filter=None,
                    split_filter=None) -> dict:
    """
    Berechnet TP/FP/FN/TN, Precision, Recall, F1.
    label_filter : 'praep', 'simple' oder None (alle)
    split_filter : 'held_out' → nur gold_split=test + FVG=0-Zeilen
                   None       → alle Zeilen
    """
    subset = results
    if label_filter == "praep":
        subset = [r for r in subset if r.get("praep", "").strip()]
    elif label_filter == "simple":
        subset = [r for r in subset if not r.get("praep", "").strip()]

    if split_filter == "held_out":
        # Nur Zeilen, die NICHT im Gold-Trainings-Split waren
        subset = [r for r in subset
                  if r.get("gold_split", "").strip() != "train"]

    # Nur Zeilen mit eindeutigem Gold-Label (0 oder 1)
    subset = [r for r in subset if r["manuell_FVG"] in ("0", "1")]

    tp = sum(1 for r in subset if r["model_pred"] == "1" and r["manuell_FVG"] == "1")
    fp = sum(1 for r in subset if r["model_pred"] == "1" and r["manuell_FVG"] == "0")
    fn = sum(1 for r in subset if r["model_pred"] == "0" and r["manuell_FVG"] == "1")
    tn = sum(1 for r in subset if r["model_pred"] == "0" and r["manuell_FVG"] == "0")

    prec   = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0

    return {
        "n": len(subset), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": recall, "f1": f1,
    }


def print_metrics(label: str, m: dict):
    print(f"\n{'─'*50}")
    print(f"  {label}  (n={m['n']})")
    print(f"{'─'*50}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
    print(f"  Precision : {m['precision']:.3f}")
    print(f"  Recall    : {m['recall']:.3f}")
    print(f"  F1        : {m['f1']:.3f}")


def main():
    fieldnames, rows = load_csv(EVAL_CSV)

    # Nur Zeilen mit manuellem Gold-Label
    eval_rows = [r for r in rows if r["manuell_FVG"].strip() in ("0", "1", "?")]
    print(f"Sätze gesamt: {len(rows)}, davon annotiert: {len(eval_rows)}")

    device = 0 if torch.cuda.is_available() else -1
    print(f"Lade Modell {MODEL_ID} (device={'GPU' if device == 0 else 'CPU'}) …")
    pipe = pipeline(
        "token-classification",
        model=MODEL_ID,
        aggregation_strategy="first",
        device=device,
    )
    print("Modell geladen.")

    print("Lade spaCy de_core_news_lg …")
    nlp = spacy.load("de_core_news_lg", disable=["ner", "textcat"])
    print("spaCy geladen.")

    # Inferenz in Batches
    sentences = [r["sentence"] for r in eval_rows]
    preds_all = []
    print(f"Führe BERT-Inferenz auf {len(sentences)} Sätzen durch …")
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i + BATCH_SIZE]
        preds_all.extend(pipe(batch))
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  {min(i + BATCH_SIZE, len(sentences))} / {len(sentences)}", end="\r")
    print(f"  {len(sentences)} / {len(sentences)} ✓")

    # spaCy-Batch-Verarbeitung
    print("Führe syntaktische Filterung durch …")
    spacy_docs = list(nlp.pipe(sentences, batch_size=64))
    filtered_preds = [
        syntactic_filter(sent, ents, nlp)
        for sent, ents, doc in zip(sentences, preds_all, spacy_docs)
        # Doc wird hier via nlp.pipe gecacht — syntactic_filter ruft nlp(text) intern auf
        # Daher direkt über den Filter iterieren:
    ]
    # Effizienter: Doc direkt übergeben (Filter leicht anpassen)
    # Einfacherer Weg: syntactic_filter nutzt nlp(text) → bereits gecacht durch nlp.pipe
    filtered_preds = [syntactic_filter(sent, ents, nlp)
                      for sent, ents in zip(sentences, preds_all)]
    print(f"  {len(filtered_preds)} Sätze gefiltert ✓")

    # Auswertung pro Zeile
    results = []
    for row, entities in zip(eval_rows, filtered_preds):
        verb_tok  = row.get("verb_token", "").strip()
        noun_tok  = row.get("noun_token", "").strip()
        praep_tok = row.get("praep", "").strip()

        pred = model_predicts_fvg(
            entities, row["sentence"], verb_tok, noun_tok, praep_tok
        )
        row["model_pred"]  = "1" if pred else "0"
        row["model_score"] = ""
        results.append(row)

    # Ergebnisse speichern
    extra = ["model_pred", "model_score"]
    out_fieldnames = fieldnames + [c for c in extra if c not in fieldnames]

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, delimiter=";",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nErgebnisse gespeichert: {OUT_CSV.name}")

    # Metriken ausgeben
    from collections import Counter

    print("\n" + "═" * 54)
    print("  MODELL-EVALUATION — Held-out Test-Split (unverzerrt)")
    print("═" * 54)
    print_metrics("Alle Paare",          compute_metrics(results, split_filter="held_out"))
    print_metrics("Einfache Verb+Nomen", compute_metrics(results, "simple", "held_out"))
    print_metrics("Präpositions-FVG",    compute_metrics(results, "praep",  "held_out"))

    print("\n" + "─" * 54)
    print("  (Alle Paare inkl. Gold-Train — nur als Referenz)")
    print("─" * 54)
    print_metrics("Alle Paare (gesamt)", compute_metrics(results))

    # Fehler-Analyse auf Held-out
    held_out = [r for r in results if r.get("gold_split", "").strip() != "train"
                and r["manuell_FVG"] in ("0", "1")]

    fn_rows = [r for r in held_out if r["model_pred"] == "0" and r["manuell_FVG"] == "1"]
    fn_verbs = Counter(r["verb_lemma"] for r in fn_rows)
    print("\n  Verben mit den meisten False Negatives (held-out):")
    for v, c in fn_verbs.most_common(10):
        print(f"    {v}: {c}×")

    fp_rows = [r for r in held_out if r["model_pred"] == "1" and r["manuell_FVG"] == "0"]
    fp_pairs = Counter((r["verb_lemma"], r["noun_lemma"]) for r in fp_rows)
    print("\n  Häufigste False-Positive-Paare (held-out):")
    for (v, n), c in fp_pairs.most_common(8):
        print(f"    {v} + {n}: {c}×")


if __name__ == "__main__":
    main()
