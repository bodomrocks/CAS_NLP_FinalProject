#!/usr/bin/env python3
"""
Schritt 5: Evaluationssätze für manuelle Annotation extrahieren.

Liest verb_noun_fvg.csv (tabulatorgetrennt) und sucht im annotierten Test-
und Dev-Korpus nach Sätzen, die das jeweilige Verb-Nomen-Paar enthalten.
Gibt eine CSV aus, die manuell annotiert werden kann.

Ausgabe: data/eval_sentences.csv
  - sentence:        Satztext
  - verb_lemma:      Infinitiv des Verbs
  - noun_lemma:      Nomen-Lemma
  - fvg_csv:         Label aus verb_noun_fvg.csv (1 = FVG, 0 = kein FVG)
  - distant_label:   Distant-Supervision-Label (FVG / kein FVG / nicht gefunden)
  - verb_token:      Gefundenes Verb-Token im Satz
  - noun_token:      Gefundenes Nomen-Token im Satz
  - manuell_FVG:     (leer – für manuelle Annotation)
  - anmerkungen:     (leer – für Anmerkungen)
"""

import csv
import json
import random
from pathlib import Path
from collections import defaultdict

import spacy
from tqdm import tqdm

BASE_DIR   = Path(__file__).resolve().parent.parent
CSV_FILE   = BASE_DIR / "data" / "verb_noun_fvg.csv"
OUT_FILE   = BASE_DIR / "data" / "eval_sentences.csv"

# Quellen: zuerst Test, dann Dev (Train nur als Fallback)
SOURCES = [
    ("test",  BASE_DIR / "data" / "annotated_test.jsonl"),
    ("dev",   BASE_DIR / "data" / "annotated_dev.jsonl"),
    ("train", BASE_DIR / "data" / "annotated_train.jsonl"),
]

MAX_PER_PAIR   = 3   # max. Sätze pro Verb-Nomen-Paar
RANDOM_SEED    = 42
MIN_SENT_LEN   = 8   # Mindestlänge (Tokens) für Satzqualität
MAX_SENT_LEN   = 60  # Maximallänge für Lesbarkeit


def load_pairs(csv_path: Path) -> list[tuple[str, str, int]]:
    """Liest verb_noun_fvg.csv (Tab-getrennt). Gibt [(verb, noun, fvg)] zurück."""
    pairs = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            verb  = row["verb_lemma"].strip().lower()
            noun  = row["noun_lemma"].strip()        # Großschreibung beibehalten
            label = int(row["FVG"].strip())
            pairs.append((verb, noun, label))
    print(f"  {len(pairs)} Verb-Nomen-Paare geladen.")
    return pairs


def load_sentences(jsonl_path: Path) -> list[dict]:
    sents = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sents.append(json.loads(line))
    return sents


def has_fvg_label(labels: list[str]) -> bool:
    return any(l != "O" for l in labels)


def find_matching_sentences(
    pairs: list[tuple[str, str, int]],
    all_sents: list[tuple[str, dict]],   # (source, sent_dict)
    nlp,
) -> dict[int, list[dict]]:
    """
    Für jedes Paar: finde Sätze, die sowohl das Nomen (Oberflächenform)
    als auch das Verb (lemmatisiert) enthalten.

    Strategie:
      1. Vorfilterung per Nomen-Oberflächenform (schnell, ohne spaCy)
      2. Lemmatisierung der Kandidatensätze per spaCy
      3. Prüfung, ob Verb-Lemma im Satz vorkommt
    """
    # Index: noun_lemma → Liste von Paar-Indizes
    noun_to_pairs: dict[str, list[int]] = defaultdict(list)
    for i, (verb, noun, _) in enumerate(pairs):
        noun_to_pairs[noun].append(i)

    # Vorfilterung: Sätze, die das Nomen enthalten
    # (Nomen sind im Deutschen großgeschrieben → Direktvergleich möglich)
    candidates_by_pair: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for source, sent in all_sents:
        token_set = set(sent["tokens"])
        for noun, pair_ids in noun_to_pairs.items():
            if noun in token_set:
                for pid in pair_ids:
                    candidates_by_pair[pid].append((source, sent))

    print(f"  Vorfilterung: {sum(len(v) for v in candidates_by_pair.values())} Kandidaten "
          f"für {len(candidates_by_pair)} Paare.")

    # Lemmatisierung der Kandidatensätze und Verb-Prüfung
    results: dict[int, list[dict]] = defaultdict(list)
    pair_verbs = {i: verb for i, (verb, _, _) in enumerate(pairs)}

    # Alle einzigartigen Kandidatensätze sammeln (ein Satz kann für mehrere Paare relevant sein)
    all_candidate_texts: dict[int, tuple[str, dict]] = {}  # sent_id → (source, sent)
    sent_to_pairs: dict[int, list[int]] = defaultdict(list)  # sent_id → [pair_ids]
    for pid, sents in candidates_by_pair.items():
        for source, sent in sents:
            sid = id(sent)  # Python-Objekt-ID als eindeutige Kennung
            all_candidate_texts[sid] = (source, sent)
            sent_to_pairs[sid].append(pid)

    print(f"  Lemmatisiere {len(all_candidate_texts)} einzigartige Kandidatensätze …")

    # Batch-Verarbeitung mit spaCy
    sid_list  = list(all_candidate_texts.keys())
    text_list = [" ".join(all_candidate_texts[sid][1]["tokens"]) for sid in sid_list]

    for idx, doc in enumerate(tqdm(
        nlp.pipe(text_list, batch_size=64),
        total=len(text_list), desc="spaCy-Lemmatisierung"
    )):
        sid      = sid_list[idx]
        source, sent = all_candidate_texts[sid]
        tokens   = sent["tokens"]
        labels   = sent["labels"]
        n_tokens = len(tokens)

        # Lemma-Menge des Satzes
        lemma_to_tokens: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for t in doc:
            if not t.is_space:
                lemma_to_tokens[t.lemma_.lower()].append((t.i, t.text))

        for pid in sent_to_pairs[sid]:
            if len(results[pid]) >= MAX_PER_PAIR:
                continue
            verb, noun, csv_label = pairs[pid]
            if verb not in lemma_to_tokens:
                continue
            # Längenfilter
            if not (MIN_SENT_LEN <= n_tokens <= MAX_SENT_LEN):
                continue

            verb_tokens  = lemma_to_tokens[verb]
            # Nomen-Token finden (Oberflächenform)
            noun_tok_idx = [i for i, tok in enumerate(tokens) if tok == noun]
            if not noun_tok_idx:
                continue

            # Prüfen: ist das Nomen dieses Paars konkret als B-NOM annotiert?
            noun_is_nom = any(
                labels[i] in ("B-NOM", "I-NOM")
                for i in noun_tok_idx
                if i < len(labels)
            )
            # Ist das Verb als B-VERB annotiert?
            verb_tok_positions = [
                j for j, tok in enumerate(tokens)
                if tok in {vt for _, vt in verb_tokens}
                and j < len(labels)
                and labels[j] == "B-VERB"
            ]
            ds_label = "FVG" if (noun_is_nom and verb_tok_positions) else "kein FVG"

            results[pid].append({
                "source":        source,
                "sentence":      " ".join(tokens),
                "verb_lemma":    verb,
                "noun_lemma":    noun,
                "fvg_csv":       csv_label,
                "distant_label": ds_label,
                "verb_token":    verb_tokens[0][1] if verb_tokens else "",
                "noun_token":    noun,
            })

    return results


def main():
    random.seed(RANDOM_SEED)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Lade spaCy …")
    nlp = spacy.load("de_core_news_lg", disable=["ner", "textcat"])

    print("Lade Verb-Nomen-Paare …")
    pairs = load_pairs(CSV_FILE)

    # Sätze aus allen Quellen laden
    print("Lade Sätze aus Test / Dev / Train …")
    all_sents: list[tuple[str, dict]] = []
    for source_name, path in SOURCES:
        sents = load_sentences(path)
        all_sents.extend((source_name, s) for s in sents)
        print(f"  {len(sents):,} Sätze aus {source_name}")

    # Matching
    print("\nSuche Sätze für jedes Verb-Nomen-Paar …")
    results = find_matching_sentences(pairs, all_sents, nlp)

    # Ausgabe-CSV schreiben
    fieldnames = [
        "paar_nr", "verb_lemma", "noun_lemma", "fvg_csv",
        "distant_label", "verb_token", "noun_token",
        "sentence", "source",
        "manuell_FVG", "anmerkungen",
    ]

    rows = []
    no_match = []

    for i, (verb, noun, csv_label) in enumerate(pairs):
        pair_results = results.get(i, [])
        if not pair_results:
            no_match.append(f"{verb} + {noun}")
            continue
        random.shuffle(pair_results)
        for r in pair_results[:MAX_PER_PAIR]:
            rows.append({
                "paar_nr":      i + 1,
                "verb_lemma":   verb,
                "noun_lemma":   noun,
                "fvg_csv":      csv_label,
                "distant_label": r["distant_label"],
                "verb_token":   r["verb_token"],
                "noun_token":   r["noun_token"],
                "sentence":     r["sentence"],
                "source":       r["source"],
                "manuell_FVG":  "",
                "anmerkungen":  "",
            })

    with open(OUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ {len(rows)} Sätze für {len(rows) - len(no_match)} / {len(pairs)} Paare")
    print(f"  → {OUT_FILE}")

    if no_match:
        print(f"\n{len(no_match)} Paare ohne Treffer im Korpus:")
        for p in no_match:
            print(f"  – {p}")


if __name__ == "__main__":
    main()
