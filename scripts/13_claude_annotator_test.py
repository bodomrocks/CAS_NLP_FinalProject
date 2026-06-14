#!/usr/bin/env python3
"""
Schritt 13: Claude als Annotator testen — Übereinstimmung mit Gold-Labels.

Lädt pair_test.jsonl (244 manuell annotierte Paare) und lässt
claude-haiku-4-5-20251001 für jedes Paar entscheiden, ob ein FVG vorliegt.
Vergleicht die Claude-Antworten mit den manuellen Gold-Labels.

Wenn die Übereinstimmung gut genug ist (F1 >= 0.75), können wir Claude
anschliessend für die ~37.000 DS-Paare aus annotated_train.jsonl einsetzen.

Verwendung:
  python 13_claude_annotator_test.py
  python 13_claude_annotator_test.py --out data/claude_test_results.csv
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import anthropic

TEST_JSONL = Path(__file__).resolve().parent.parent / "data" / "pair_test.jsonl"
OUT_CSV    = Path(__file__).resolve().parent.parent / "data" / "claude_test_results_v3.csv"

SYSTEM_PROMPT = """\
Du bist Linguist und Experte für deutsche Syntax, spezialisiert auf \
Funktionsverbgefüge (FVG) in juristischen Texten.

Ein FVG liegt vor, wenn das Verb semantisch abgeschwächt ist und das Nomen den \
eigentlichen Inhalt trägt. Zwei Tests helfen bei der Entscheidung:

TEST 1 — SIMPLEX-TEST (wichtigstes Kriterium):
Gibt es ein morphologisch verwandtes einfaches Verb, das dieselbe Bedeutung hat?
→ JA → starkes Indiz für FVG
  Beispiele: "Baubewilligung erteilen" ↔ bewilligen
             "Anweisung erteilen" ↔ anweisen
             "Antrag stellen" ↔ beantragen
             "Kündigung aussprechen" ↔ kündigen
             "Verordnung erlassen" ↔ verordnen
             "Schätzung vornehmen" ↔ schätzen

TEST 2 — SELEKTIONSTEST:
Kann das Verb in derselben Konstruktion mit sehr vielen anderen Nomen stehen \
und behält dabei seine volle Bedeutung?
→ JA (sehr produktiv) → eher KEIN FVG
  Beispiele: "eine Untersuchung führen" (führen = leiten, viele Nomen möglich)
             "Abgaben erheben" (erheben = einziehen, noch voll lexikalisch)
             "eine Eingabe einreichen" (einreichen = produktiv, nicht abgeschwächt)

Klare Funktionsverben: erheben, stellen, treffen, nehmen, fassen, ziehen, bringen,
  setzen, leisten, vornehmen, aussprechen, treten, lassen, erlassen, erteilen
  (erteilen ist produktiv, aber wenn ein Simplex existiert → FVG)

Klare FVG-Beispiele:
  "Beschwerde erheben", "Antrag stellen", "Massnahme treffen", "Abstand nehmen"
  "Kündigung aussprechen", "Entscheid fassen", "Folgerung ziehen"
  "Ausnahme machen", "Vorzug geben", "Begründung geben", "Zustimmung geben"
  "Beurteilung vornehmen", "Abwägung vornehmen", "Schätzung vornehmen"
  "Baubewilligung erteilen", "Anweisung erteilen", "Verordnung erlassen"
  "in Kraft treten", "ausser Kraft setzen", "in Frage stellen"
  "zu Grunde legen", "in Wiedererwägung ziehen", "ausser Acht lassen"
  "in Einklang bringen", "zu Stande kommen", "in Betracht fallen/ziehen"
  "ausser Betracht lassen", "mit Strafe bedrohen"

Kein FVG:
  "eine Untersuchung/Strafuntersuchung führen" — führen zu produktiv, voll lexikalisch
  "Abgaben erheben" — erheben = einziehen, kein Simplex im gleichen Sinne
  "von Amtes wegen" + Verb — "von Amtes wegen" ist Adverbial, kein FVG-Nomen
  "auf Abweisung schliessen/erkennen" — fester Rechtsausdruck, kein FVG
  "eine Bedingung stellen" — kein Simplex; stellen hat noch volle Bedeutung
  "eine Eingabe einreichen" — einreichen zu produktiv, nicht abgeschwächt

Gefragt wird immer: Bildet das ANGEGEBENE Verb mit dem ANGEGEBENEN Nomen \
(in diesem konkreten Satz) ein FVG?

Antworte ausschliesslich mit "ja" oder "nein".\
"""

USER_TEMPLATE = """\
Satz: {sentence}

Verb: {verb_token}
Nomen/Muster: {pattern}

Bilden diese beiden in diesem Satz ein Funktionsverbgefüge?\
"""


def ask_claude(client: anthropic.Anthropic, sentence: str,
               verb_token: str, pattern: str) -> str:
    """Fragt Claude und gibt 'ja', 'nein' oder '?' zurück."""
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": USER_TEMPLATE.format(
                sentence=sentence,
                verb_token=verb_token,
                pattern=pattern,
            ),
        }],
    )
    answer = msg.content[0].text.strip().lower()
    if answer.startswith("ja"):
        return "ja"
    if answer.startswith("nein"):
        return "nein"
    return "?"


def compute_metrics(results: list[dict]) -> dict:
    valid = [r for r in results if r["claude_label"] in ("0", "1")
             and r["gold_label"] in ("0", "1")]
    tp = sum(1 for r in valid if r["claude_label"] == "1" and r["gold_label"] == "1")
    fp = sum(1 for r in valid if r["claude_label"] == "1" and r["gold_label"] == "0")
    fn = sum(1 for r in valid if r["claude_label"] == "0" and r["gold_label"] == "1")
    tn = sum(1 for r in valid if r["claude_label"] == "0" and r["gold_label"] == "0")
    prec   = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
    acc    = (tp + tn) / len(valid) if valid else 0.0
    return dict(n=len(valid), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=prec, recall=recall, f1=f1, accuracy=acc)


def main(out_path: Path = OUT_CSV):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY nicht gesetzt.")

    client = anthropic.Anthropic(api_key=api_key)

    with open(TEST_JSONL, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    print(f"Test-Paare: {len(records)}  "
          f"({sum(r['label'] for r in records)} positiv, "
          f"{sum(1-r['label'] for r in records)} negativ)")
    print(f"Modell: claude-haiku-4-5-20251001\n")

    results = []
    n_ja = n_nein = n_unbekannt = 0

    for i, rec in enumerate(records):
        praep    = rec.get("praep", "").strip()
        noun_tok = rec.get("noun_token", "").strip()
        verb_tok = rec.get("verb_token", "").strip()
        pattern  = f"{praep} {noun_tok}".strip() if praep else noun_tok

        answer = ask_claude(client, rec["sentence"], verb_tok, pattern)

        claude_label = "1" if answer == "ja" else ("0" if answer == "nein" else "?")
        gold_label   = str(rec["label"])

        n_ja       += answer == "ja"
        n_nein     += answer == "nein"
        n_unbekannt += answer == "?"

        correct = "✓" if claude_label == gold_label else "✗"
        print(f"[{i+1:3}/{len(records)}] {correct} Claude={answer:4}  Gold={gold_label}  "
              f"{verb_tok:15} + {pattern[:20]}")

        results.append({
            "sentence"    : rec["sentence"],
            "verb_token"  : verb_tok,
            "noun_token"  : noun_tok,
            "praep"       : praep,
            "pattern"     : pattern,
            "verb_lemma"  : rec.get("verb_lemma", ""),
            "noun_lemma"  : rec.get("noun_lemma", ""),
            "gold_label"  : gold_label,
            "claude_answer": answer,
            "claude_label": claude_label,
        })

        # Kleine Pause um Rate-Limits zu vermeiden
        if (i + 1) % 50 == 0:
            time.sleep(1)

    # Ergebnisse speichern
    fieldnames = list(results[0].keys())
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        w.writeheader()
        w.writerows(results)
    print(f"\nErgebnisse gespeichert: {out_path.name}")

    # Metriken
    m = compute_metrics(results)
    print(f"\n{'═'*50}")
    print(f"  ÜBEREINSTIMMUNG Claude vs. Gold  (n={m['n']})")
    print(f"{'═'*50}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
    print(f"  Precision : {m['precision']:.3f}")
    print(f"  Recall    : {m['recall']:.3f}")
    print(f"  F1        : {m['f1']:.3f}")
    print(f"  Accuracy  : {m['accuracy']:.3f}")
    print(f"\n  Antwortverteilung: ja={n_ja}, nein={n_nein}, ?={n_unbekannt}")

    if m['f1'] >= 0.75:
        print("\n  → Übereinstimmung ausreichend für DS-Annotation.")
    else:
        print("\n  → Übereinstimmung zu niedrig. Prompt anpassen.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()
    main(Path(args.out))
