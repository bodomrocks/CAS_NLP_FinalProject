#!/usr/bin/env python3
"""
Schritt 14: Claude annotiert DS-Paare aus annotated_train.jsonl.

Für jedes Paar mit B-VERB + B-NOM fragt das Skript claude-haiku-4-5-20251001,
ob ein FVG vorliegt. Der System-Prompt wird per Prompt-Caching gecacht
(~77% Kosteneinsparung gegenüber unkachierter Variante).

Ausgabe: data/ds_claude_labels.jsonl
  Felder: sentence, verb_token, noun_token, praep, claude_answer, claude_label

Verwendung:
  python 14_claude_annotate_ds.py
  python 14_claude_annotate_ds.py --max_pairs 2000   # Testlauf
  python 14_claude_annotate_ds.py --resume           # Fortsetzen nach Abbruch
"""

import argparse
import json
import os
import time
from pathlib import Path

import anthropic

ANNOTATED_TRAIN = Path(__file__).resolve().parent.parent / "data" / "annotated_train.jsonl"
OUT_JSONL       = Path(__file__).resolve().parent.parent / "data" / "ds_claude_labels.jsonl"

BATCH_SIZE   = 100   # Speichern alle N Anfragen
RETRY_DELAY  = 5     # Sekunden Wartezeit bei Rate-Limit-Fehler
MAX_RETRIES  = 6

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
  "eine Untersuchung/Strafuntersuchung führen" — führen zu produktiv
  "Abgaben erheben" — erheben = einziehen, kein Simplex im gleichen Sinne
  "von Amtes wegen" + Verb — Adverbial, kein FVG-Nomen
  "auf Abweisung schliessen/erkennen" — fester Rechtsausdruck, kein FVG
  "eine Bedingung stellen" — kein Simplex; stellen hat noch volle Bedeutung
  "eine Eingabe einreichen" — einreichen zu produktiv

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


def extract_pairs(path: Path) -> list[dict]:
    """Extrahiert alle Verb+Nomen-Paare mit B-VERB + B-NOM aus BIO-Daten."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "B-VERB" not in r["labels"] or "B-NOM" not in r["labels"]:
                continue
            vi = r["labels"].index("B-VERB")
            ni = r["labels"].index("B-NOM")
            # Vollständige NOM-Span (B-NOM + I-NOM)
            nom_tokens = [r["tokens"][ni]]
            j = ni + 1
            while j < len(r["labels"]) and r["labels"][j] == "I-NOM":
                nom_tokens.append(r["tokens"][j])
                j += 1
            # Präposition: erstes Token der NOM-Span wenn klein geschrieben
            first = nom_tokens[0]
            praep = first if first.islower() and len(first) <= 5 else ""
            noun  = nom_tokens[-1] if praep else nom_tokens[0]
            pairs.append({
                "sentence"  : " ".join(r["tokens"]),
                "verb_token": r["tokens"][vi],
                "noun_token": noun,
                "praep"     : praep,
                "nom_span"  : " ".join(nom_tokens),
            })
    return pairs


def ask_claude(client: anthropic.Anthropic, sentence: str,
               verb_token: str, pattern: str,
               retries: int = MAX_RETRIES) -> str:
    """Fragt Claude mit gecachtem System-Prompt; gibt 'ja', 'nein' oder '?' zurück."""
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=5,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
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
        except anthropic.RateLimitError:
            wait = RETRY_DELAY * (2 ** attempt)
            print(f"\n  Rate-Limit — warte {wait}s …", end="", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"\n  Fehler: {e} — warte {RETRY_DELAY}s …", end="", flush=True)
            time.sleep(RETRY_DELAY)
    return "?"


def load_done(path: Path) -> set[str]:
    """Lädt bereits annotierte Sätze (für Resume-Modus)."""
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add(r["sentence"] + "|" + r["verb_token"] + "|" + r["noun_token"])
    return done


def main(max_pairs: int | None, resume: bool):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY nicht gesetzt.")

    client = anthropic.Anthropic(api_key=api_key)

    print("Extrahiere DS-Paare …")
    pairs = extract_pairs(ANNOTATED_TRAIN)
    print(f"  {len(pairs):,} Paare gefunden.")

    if max_pairs:
        pairs = pairs[:max_pairs]
        print(f"  Begrenzung auf {max_pairs} Paare (--max_pairs).")

    done = load_done(OUT_JSONL) if resume else set()
    if done:
        print(f"  {len(done)} bereits annotiert — werden übersprungen.")

    mode = "a" if resume and OUT_JSONL.exists() else "w"
    out_f = open(OUT_JSONL, mode, encoding="utf-8")

    n_ja = n_nein = n_skip = n_unknown = 0
    buffer = []
    t0 = time.time()

    for i, pair in enumerate(pairs):
        key = pair["sentence"] + "|" + pair["verb_token"] + "|" + pair["noun_token"]
        if key in done:
            n_skip += 1
            continue

        praep   = pair["praep"]
        pattern = f"{praep} {pair['noun_token']}".strip() if praep else pair["noun_token"]

        answer = ask_claude(client, pair["sentence"], pair["verb_token"], pattern)
        label  = "1" if answer == "ja" else ("0" if answer == "nein" else "?")

        n_ja      += answer == "ja"
        n_nein    += answer == "nein"
        n_unknown += answer == "?"

        record = {
            "sentence"    : pair["sentence"],
            "verb_token"  : pair["verb_token"],
            "noun_token"  : pair["noun_token"],
            "praep"       : praep,
            "nom_span"    : pair["nom_span"],
            "claude_answer": answer,
            "claude_label": label,
        }
        buffer.append(json.dumps(record, ensure_ascii=False))

        if len(buffer) >= BATCH_SIZE:
            out_f.write("\n".join(buffer) + "\n")
            out_f.flush()
            buffer.clear()

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1 - n_skip) / elapsed
            remain  = (len(pairs) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:5}/{len(pairs)}]  ja={n_ja}  nein={n_nein}  "
                  f"?={n_unknown}  "
                  f"({rate:.1f} req/s, ~{remain/60:.0f} Min. verbleibend)", flush=True)

    if buffer:
        out_f.write("\n".join(buffer) + "\n")
    out_f.close()

    total = n_ja + n_nein + n_unknown
    print(f"\nFertig. {total} annotiert ({n_skip} übersprungen).")
    print(f"  ja={n_ja} ({n_ja/total*100:.1f}%)  "
          f"nein={n_nein} ({n_nein/total*100:.1f}%)  "
          f"?={n_unknown}")
    print(f"Ausgabe: {OUT_JSONL}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Fortsetzen (bereits annotierte Paare überspringen)")
    args = ap.parse_args()
    main(args.max_pairs, args.resume)
