#!/usr/bin/env python3
"""
Schritt 5b: Präpositionale FVG aus praep_noun_verb.csv extrahieren.

Parst die Datei, dedupliziert Einträge, sucht Sätze im Korpus und
hängt sie an data/eval_sentences.csv an (mit leerer fvg_csv-Spalte,
da noch kein Label vorhanden).

Format praep_noun_verb.csv:
  "   938 in Frage kommen"
  "   278 Grunde legen"       ← ohne explizite Präp
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import spacy
from tqdm import tqdm

BASE_DIR   = Path(__file__).resolve().parent.parent
PRAEP_FILE = BASE_DIR / "data" / "praep_noun_verb.csv"
EVAL_CSV   = BASE_DIR / "data" / "eval_sentences.csv"
SOURCES    = [
    ("test",  BASE_DIR / "data" / "annotated_test.jsonl"),
    ("dev",   BASE_DIR / "data" / "annotated_dev.jsonl"),
    ("train", BASE_DIR / "data" / "annotated_train.jsonl"),
]

MAX_PER_PATTERN = 3
MIN_SENT_LEN    = 8
MAX_SENT_LEN    = 60
RANDOM_SEED     = 42

# Bekannte deutsche Präpositionen (Kleinschreibung)
PREPS = {
    "in", "an", "auf", "zu", "von", "aus", "bei", "mit", "nach", "über",
    "unter", "vor", "hinter", "neben", "zwischen", "ausser", "nebst",
    "wegen", "trotz", "während", "gegen", "durch", "für", "ohne",
}


def parse_patterns(path: Path) -> list[dict]:
    """
    Liest praep_noun_verb.csv und gibt deduplizierte Muster zurück.
    Rückgabe: [{"full": str, "praep": str, "noun": str, "verb": str, "count": int}]
    """
    seen: dict[str, int] = {}   # text_lower → count (nimm höchsten)
    raw: dict[str, dict] = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            count    = int(parts[0])
            fvg_text = parts[1].strip()
            key      = fvg_text.lower()
            if key in seen:
                seen[key] = max(seen[key], count)
            else:
                seen[key] = count
                raw[key]  = fvg_text   # Originalform (Großschreibung)

    patterns = []
    for key, fvg_text in raw.items():
        tokens = fvg_text.split()
        if len(tokens) < 2:
            continue
        verb  = tokens[-1].lower()
        rest  = tokens[:-1]
        # Erste Token: Präposition wenn kleingeschrieben und in PREPS
        if rest and rest[0].lower() in PREPS:
            praep = rest[0].lower()
            nouns = rest[1:]
        else:
            praep = ""
            nouns = rest
        # Nomen: erstes echtes Nomen in der Rest-Liste (gross)
        noun = " ".join(nouns)   # kann mehrere Tokens sein, z.B. "Amtes wegen"
        patterns.append({
            "full":  fvg_text,
            "praep": praep,
            "noun":  noun,          # z.B. "Frage", "Betracht", "Amtes wegen"
            "verb":  verb,
            "count": seen[key],
        })

    patterns.sort(key=lambda x: -x["count"])
    print(f"  {len(patterns)} deduplizierte Muster")
    return patterns


def load_sentences(jsonl_path: Path) -> list[dict]:
    sents = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sents.append(json.loads(line))
    return sents


def find_sentences(patterns: list[dict], all_sents: list[tuple], nlp) -> dict[int, list[dict]]:
    """
    Für jedes Muster: finde Sätze, die Nomen (Oberfläche) + Verb (Lemma) enthalten.
    """
    # Index: erstes Nomen-Token → Pattern-IDs
    first_noun_to_pats: dict[str, list[int]] = defaultdict(list)
    for i, pat in enumerate(patterns):
        first_noun = pat["noun"].split()[0]   # z.B. "Frage", "Amtes"
        first_noun_to_pats[first_noun].append(i)

    # Vorfilterung per Nomen-Oberflächenform
    candidates_by_pat: dict[int, list[tuple]] = defaultdict(list)
    for source, sent in all_sents:
        token_set = set(sent["tokens"])
        for noun_tok, pat_ids in first_noun_to_pats.items():
            if noun_tok in token_set:
                for pid in pat_ids:
                    candidates_by_pat[pid].append((source, sent))

    print(f"  Vorfilterung: {sum(len(v) for v in candidates_by_pat.values())} Kandidaten "
          f"für {len(candidates_by_pat)} Muster")

    # Einzigartige Kandidatensätze sammeln
    all_cand: dict[int, tuple] = {}
    sent_to_pats: dict[int, list[int]] = defaultdict(list)
    for pid, sents in candidates_by_pat.items():
        for source, sent in sents:
            sid = id(sent)
            all_cand[sid] = (source, sent)
            sent_to_pats[sid].append(pid)

    print(f"  Lemmatisiere {len(all_cand)} Kandidatensätze …")
    sid_list  = list(all_cand.keys())
    text_list = [" ".join(all_cand[sid][1]["tokens"]) for sid in sid_list]

    results: dict[int, list[dict]] = defaultdict(list)

    for idx, doc in enumerate(tqdm(nlp.pipe(text_list, batch_size=64),
                                   total=len(text_list), desc="spaCy")):
        sid = sid_list[idx]
        source, sent = all_cand[sid]
        tokens = sent["tokens"]
        n_tok  = len(tokens)

        if not (MIN_SENT_LEN <= n_tok <= MAX_SENT_LEN):
            continue

        lemma_to_surfs: dict[str, list[str]] = defaultdict(list)
        for t in doc:
            if not t.is_space:
                lemma_to_surfs[t.lemma_.lower()].append(t.text)

        for pid in sent_to_pats[sid]:
            if len(results[pid]) >= MAX_PER_PATTERN:
                continue
            pat = patterns[pid]
            verb_lemma  = pat["verb"]
            first_noun  = pat["noun"].split()[0]
            praep       = pat["praep"]

            if verb_lemma not in lemma_to_surfs:
                continue
            if first_noun not in tokens:
                continue
            if praep and praep not in [t.lower() for t in tokens]:
                continue

            verb_surf = lemma_to_surfs[verb_lemma][0]

            results[pid].append({
                "source":        source,
                "sentence":      " ".join(tokens),
                "verb_lemma":    verb_lemma,
                "noun_lemma":    first_noun,
                "praep":         praep,
                "full_pattern":  pat["full"],
                "fvg_csv":       "",          # kein Vorab-Label
                "distant_label": "",
                "verb_token":    verb_surf,
                "noun_token":    first_noun,
            })

    return results


def append_to_csv(eval_csv: Path, new_rows: list[dict], existing_fieldnames: list[str]):
    """Hängt neue Zeilen an eval_sentences.csv an (fügt praep-Spalte hinzu falls nötig)."""
    # Bestehende CSV lesen
    existing = []
    with open(eval_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = list(reader.fieldnames or existing_fieldnames)
        existing   = list(reader)

    # Spalten ergänzen falls nötig
    for col in ("praep", "full_pattern"):
        if col not in fieldnames:
            fieldnames.append(col)
            for r in existing:
                r[col] = ""

    # Neue Zeilen vorbereiten
    # Höchste paar_nr ermitteln
    max_nr = max((int(r.get("paar_nr", 0) or 0) for r in existing), default=0)

    prepared = []
    pair_counter = max_nr
    seen_pairs: set[tuple] = set()
    for row in new_rows:
        key = (row["verb_lemma"], row["noun_lemma"], row.get("praep", ""))
        if key not in seen_pairs:
            pair_counter += 1
            seen_pairs.add(key)
        prepared.append({
            "paar_nr":       pair_counter,
            "verb_lemma":    row["verb_lemma"],
            "noun_lemma":    row["noun_lemma"],
            "fvg_csv":       row.get("fvg_csv", ""),
            "distant_label": row.get("distant_label", ""),
            "verb_token":    row.get("verb_token", ""),
            "noun_token":    row.get("noun_token", ""),
            "sentence":      row["sentence"],
            "source":        row["source"],
            "manuell_FVG":   "",
            "anmerkungen":   "",
            "praep":         row.get("praep", ""),
            "full_pattern":  row.get("full_pattern", ""),
        })

    all_rows = existing + prepared
    with open(eval_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  {len(prepared)} neue Sätze angehängt → {eval_csv.name}")
    print(f"  Gesamt: {len(all_rows)} Sätze")
    return fieldnames


def main():
    random.seed(RANDOM_SEED)

    print("Lade spaCy …")
    nlp = spacy.load("de_core_news_lg", disable=["ner", "textcat"])

    print("Parse Präp-FVG-Liste …")
    patterns = parse_patterns(PRAEP_FILE)

    print("Lade Sätze …")
    all_sents = []
    for name, path in SOURCES:
        sents = load_sentences(path)
        all_sents.extend((name, s) for s in sents)
        print(f"  {len(sents):,} Sätze aus {name}")

    print("\nSuche Sätze …")
    results = find_sentences(patterns, all_sents, nlp)

    # Ergebnisse flachlegen
    new_rows = []
    no_match = []
    for i, pat in enumerate(patterns):
        hits = results.get(i, [])
        if not hits:
            no_match.append(pat["full"])
        else:
            random.shuffle(hits)
            new_rows.extend(hits[:MAX_PER_PATTERN])

    print(f"\n{len(new_rows)} Sätze für {len(patterns) - len(no_match)} / {len(patterns)} Muster")

    print("Hänge an eval_sentences.csv an …")
    append_to_csv(EVAL_CSV, new_rows, [])

    if no_match:
        print(f"\n{len(no_match)} Muster ohne Treffer:")
        for p in no_match:
            print(f"  – {p}")


if __name__ == "__main__":
    main()
