#!/usr/bin/env python3
"""
Schritt 10: Syntaktischen Filter v2 auf gespeicherten Modellvorhersagen testen.

Liest eval_model_results.csv (bereits annotierte Vorhersagen),
wendet den verbesserten Filter auf konkrete FP-Sätze an und zeigt,
wie viele Kreuzklausel-FPs korrekt verworfen werden.

Verwendung:
  python 10_test_filter_v2.py
"""

import csv
import sys
from pathlib import Path

import spacy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from fvg_syntax_filter import _same_clause, _tokens_in_span, _is_fvg_dep

RESULTS_CSV = Path(__file__).resolve().parent.parent / "data" / "eval_model_results.csv"


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)


def find_char_offset(sentence: str, token: str) -> list[int]:
    pos, offsets = 0, []
    while True:
        idx = sentence.find(token, pos)
        if idx == -1:
            break
        before = idx == 0 or sentence[idx - 1] == " "
        after  = (idx + len(token) == len(sentence)
                  or sentence[idx + len(token)] in " .,;:!?\"'()")
        if before and after:
            offsets.append(idx)
        pos = idx + 1
    return offsets


def check_dep(sentence: str, verb_tok_str: str, noun_tok_str: str, nlp) -> bool:
    """
    Prüft direkt per Dependency-Graph, ob verb_tok und noun_tok
    eine FVG-Relation aufweisen.
    """
    doc = nlp(sentence)

    v_offsets = find_char_offset(sentence, verb_tok_str)
    n_offsets = find_char_offset(sentence, noun_tok_str)

    if not v_offsets or not n_offsets:
        return False

    # Token an der ersten gefundenen Position
    def tok_at(offsets):
        for off in offsets:
            cands = _tokens_in_span(doc, off, off + 1)
            nouns = [t for t in cands if t.pos_ in ("NOUN", "PROPN")]
            return nouns[0] if nouns else (cands[0] if cands else None)
        return None

    def verb_at(offsets):
        for off in offsets:
            cands = _tokens_in_span(doc, off, off + 1)
            verbs = [t for t in cands if t.pos_ == "VERB"]
            return verbs[0] if verbs else (cands[0] if cands else None)
        return None

    v_tok = verb_at(v_offsets)
    n_tok = tok_at(n_offsets)
    return _is_fvg_dep(v_tok, n_tok)


def main():
    print("Lade spaCy de_core_news_lg …")
    nlp = spacy.load("de_core_news_lg", disable=["ner", "textcat"])
    print("spaCy geladen.\n")

    rows = load_csv(RESULTS_CSV)

    # Nur Held-out-Zeilen mit eindeutigem Gold-Label
    held_out = [r for r in rows
                if r.get("gold_split", "").strip() != "train"
                and r["manuell_FVG"] in ("0", "1")]

    fp_rows = [r for r in held_out
               if r["model_pred"] == "1" and r["manuell_FVG"] == "0"]
    tp_rows = [r for r in held_out
               if r["model_pred"] == "1" and r["manuell_FVG"] == "1"]
    fn_rows = [r for r in held_out
               if r["model_pred"] == "0" and r["manuell_FVG"] == "1"]
    tn_rows = [r for r in held_out
               if r["model_pred"] == "0" and r["manuell_FVG"] == "0"]

    print(f"Held-out: {len(held_out)} Sätze")
    print(f"  TP={len(tp_rows)}  FP={len(fp_rows)}  FN={len(fn_rows)}  TN={len(tn_rows)}")
    prec_old = len(tp_rows) / (len(tp_rows) + len(fp_rows))
    rec_old  = len(tp_rows) / (len(tp_rows) + len(fn_rows))
    f1_old   = 2 * prec_old * rec_old / (prec_old + rec_old) if (prec_old + rec_old) else 0
    print(f"  Precision={prec_old:.3f}  Recall={rec_old:.3f}  F1={f1_old:.3f}")
    print()

    # Dependency-Check auf FPs: würde Filter v2 diese verwerfen?
    print("Prüfe FPs mit Filter v2 …")
    fp_blocked = 0
    fp_kept    = 0
    blocked_examples = []

    for r in fp_rows:
        verb_tok_str = r.get("verb_token", "").strip()
        noun_tok_str = r.get("noun_token", "").strip()
        sentence     = r.get("sentence", "").strip()
        if not verb_tok_str or not noun_tok_str or not sentence:
            fp_kept += 1
            continue

        dep_ok = check_dep(sentence, verb_tok_str, noun_tok_str, nlp)
        if dep_ok:
            fp_kept += 1
        else:
            fp_blocked += 1
            blocked_examples.append(r)

    # Dependency-Check auf TPs: würde Filter v2 diese fälschlich verwerfen?
    print("Prüfe TPs auf unbeabsichtigte Verwerfung …")
    tp_blocked = 0
    tp_kept    = 0
    for r in tp_rows:
        verb_tok_str = r.get("verb_token", "").strip()
        noun_tok_str = r.get("noun_token", "").strip()
        sentence     = r.get("sentence", "").strip()
        if not verb_tok_str or not noun_tok_str or not sentence:
            tp_kept += 1
            continue
        dep_ok = check_dep(sentence, verb_tok_str, noun_tok_str, nlp)
        if dep_ok:
            tp_kept += 1
        else:
            tp_blocked += 1

    print()
    print("═" * 54)
    print("  FILTER v2 – Simulierte Auswirkung")
    print("═" * 54)
    print(f"  FPs verworfen:         {fp_blocked} / {len(fp_rows)}  "
          f"({fp_blocked/len(fp_rows)*100:.1f}%)")
    print(f"  TPs fälschlich blockiert: {tp_blocked} / {len(tp_rows)}  "
          f"({tp_blocked/len(tp_rows)*100:.1f}%)")
    print()

    new_tp = len(tp_rows) - tp_blocked
    new_fp = len(fp_rows) - fp_blocked
    new_fn = len(fn_rows) + tp_blocked
    prec_new = new_tp / (new_tp + new_fp) if (new_tp + new_fp) else 0
    rec_new  = new_tp / (new_tp + new_fn) if (new_tp + new_fn) else 0
    f1_new   = 2 * prec_new * rec_new / (prec_new + rec_new) if (prec_new + rec_new) else 0
    print(f"  Vorher:  Prec={prec_old:.3f}  Rec={rec_old:.3f}  F1={f1_old:.3f}")
    print(f"  Nachher: Prec={prec_new:.3f}  Rec={rec_new:.3f}  F1={f1_new:.3f}")
    print()

    if blocked_examples:
        print("  Beispiele verworfener FPs:")
        for r in blocked_examples[:8]:
            print(f"    verb={r['verb_lemma']} noun={r['noun_lemma']}")
            print(f"      {r['sentence'][:110]}")
            print()


if __name__ == "__main__":
    main()
