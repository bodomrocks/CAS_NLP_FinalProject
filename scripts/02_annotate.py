#!/usr/bin/env python3
"""
Schritt 2: Automatische Annotation mit Zweipass-Strategie.

Pass 1 – Listenbasiert (hohe Präzision):
  Sucht bekannte FVG-Muster aus liste_fvg.txt im Text.

Pass 2 – Syntaxbasiert (Generalisierung):
  Für jedes bekannte Funktionsverb: sucht Nominalkomponenten anhand
  des spaCy-Dependency-Graphs. Erkennt so auch ungelistete FVG wie
  "Gehör gewähren", "Verrat üben", "Gewähr bieten".

Ausgabe: data/annotated_{train,dev,test}.jsonl
"""

import json
import random
from pathlib import Path
from typing import Iterator

import spacy
from tqdm import tqdm

BASE_DIR  = Path(__file__).resolve().parent.parent
FVG_FILE  = BASE_DIR / "data" / "liste_fvg.txt"
RAW_FILE  = Path(__file__).resolve().parent.parent / "data" / "raw_texts.jsonl"
OUT_DIR   = Path(__file__).resolve().parent.parent / "data"

WINDOW      = 20
MIN_SENT    = 5
MAX_SENT    = 200
BATCH_SIZE  = 32
RANDOM_SEED = 42

LABEL2ID = {"O": 0, "B-VERB": 1, "B-NOM": 2, "I-NOM": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ── Funktionsverb-Liste für Pass 2 ───────────────────────────────────────────
# Nur klar funktionale Verben. "haben" und "sein" sind AUSGESCHLOSSEN,
# da sie als Auxiliarverben zu viele Falschpositive erzeugen.
FUNKTIONSVERBEN = {
    # Aus der FVG-Liste (eindeutig funktional)
    "nehmen", "geben", "stellen", "setzen", "fassen", "ziehen", "bringen",
    "führen", "ergreifen", "erheben", "leisten", "machen", "treffen", "treten",
    "üben", "ausüben", "schöpfen", "anstellen", "finden", "einlegen",
    # Mit PP-Konstruktionen (zum/zur/im + Nomen)
    "kommen", "stehen", "liegen", "geraten",
    # Erweiterungen für Generalisierung (juristisch begründet)
    "bieten", "gewähren", "erlassen", "verüben", "begehen",
    "einräumen", "vorlegen", "entgegennehmen", "einnehmen",
    # Neu: häufige Funktionsverben aus manueller Evaluation (Gold-Labels)
    "vornehmen", "erteilen", "aussprechen", "belegen", "schliessen",
    "anordnen", "einreichen",
}

# Dep-Labels für Objekte und PP-Modifier (TIGER-Schema von spaCy de_core_news_lg)
OBJ_DEPS = {"oa", "obj", "dobj"}              # direktes Objekt (Akkusativ)
PP_DEPS  = {"mo", "cvc", "op", "pg", "obl"}   # PP-Modifier des Verbs

# Präpositionen, die in FVG-Nominalgruppen vorkommen (nach Lemmatisierung)
FVG_PREPS = {"in", "zu", "auf", "an", "ausser", "unter", "vor", "ohne", "aus"}

# Suffixe deverbaler/abstrakter Nomina
ABSTRACT_SUFFIXES = (
    "ung", "ion", "tion", "schaft", "heit", "keit", "tum", "nis",
    "ismus", "ität", "enz", "anz", "ur",
)

# Manuelle Ergänzung: abstrakte Nomina ohne Standardsuffix, die häufig in FVG
# auftreten, aber nicht in liste_fvg.txt stehen.
ADDITIONAL_FVG_NOUNS = {
    "verrat", "gewähr", "gehör", "kontrolle", "schutz", "pflicht",
    "obliegen", "haftung", "auskunft", "einigung", "lösung", "zweck",
    "sinn", "erfolg", "wirkung", "schaden", "nutzen", "vorteil",
    "nachteil", "risiko", "gefahr", "verantwortung", "schuld",
    "interesse", "druck", "einfluss", "kraft", "macht", "recht",
    "pflicht", "auftrag", "befehl", "mandat",
}


# ── FVG-Liste laden ────────────────────────────────────────────────────────────

def load_patterns(fvg_path: Path, nlp) -> tuple[list, set]:
    """
    Gibt (patterns, fvg_nouns) zurück:
    - patterns: [(nom_lemmas, verb_lemma, original_text)]
    - fvg_nouns: Menge aller Nominal-Lemmas aus der Liste
    """
    patterns: list = []
    fvg_nouns: set = set()

    with open(fvg_path, encoding="utf-8") as f:
        for line in f:
            # Dateiformat: entweder "FVG-Text" oder "NR\tFVG-Text"
            fvg_text = line.strip().split("\t")[-1].lower().strip()
            if not fvg_text:
                continue
            doc = nlp(fvg_text)
            tokens = [t for t in doc if not t.is_space]
            if len(tokens) < 2:
                continue

            verb_lemma = tokens[-1].lemma_.lower()
            nom_lemmas = [t.lemma_.lower() for t in tokens[:-1]]

            patterns.append((nom_lemmas, verb_lemma, fvg_text))

            # Nomina (keine Präpositionen) für FVG_NOUNS sammeln
            for t in tokens[:-1]:
                if t.pos_ not in ("ADP", "DET", "CONJ", "CCONJ", "PUNCT"):
                    fvg_nouns.add(t.lemma_.lower())

    fvg_nouns |= ADDITIONAL_FVG_NOUNS
    print(f"  {len(patterns)} FVG-Muster | {len(fvg_nouns)} FVG-Nomina")
    return patterns, fvg_nouns


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def is_fvg_noun(lemma: str, fvg_nouns: set) -> bool:
    l = lemma.lower()
    return l in fvg_nouns or any(l.endswith(s) for s in ABSTRACT_SUFFIXES)


# ── Pass 1: Listenbasierte Annotation ─────────────────────────────────────────

def pass1_list(words, lemmas, labels, patterns):
    n = len(words)
    for nom_lemmas, verb_lemma, _ in patterns:
        nom_len = len(nom_lemmas)
        for vi in range(n):
            if lemmas[vi] != verb_lemma:
                continue
            lo = max(0, vi - WINDOW)
            hi = min(n - nom_len, vi + WINDOW)
            for ni in range(lo, hi + 1):
                if ni <= vi < ni + nom_len:
                    continue
                if all(lemmas[ni + j] == nom_lemmas[j] for j in range(nom_len)):
                    if labels[vi] == "O":
                        labels[vi] = "B-VERB"
                    for j in range(nom_len):
                        if labels[ni + j] == "O":
                            labels[ni + j] = "B-NOM" if j == 0 else "I-NOM"


# ── Pass 2: Syntaxbasierte Annotation ─────────────────────────────────────────

def pass2_syntax(sent, labels, fvg_nouns):
    """
    Syntaxbasierte Generalisierung via Dependency-Graph.
    Nur präzise Abhängigkeitsrelationen: dep=oa für direkte Objekte,
    dep=mo/cvc für PP-Modifier. Subjekte (dep=sb) werden nie annotiert.
    """
    tokens = [t for t in sent if not t.is_space]
    tok_idx = {t.i: idx for idx, t in enumerate(tokens)}

    for t in tokens:
        if t.lemma_.lower() not in FUNKTIONSVERBEN:
            continue
        vi = tok_idx.get(t.i)
        if vi is None:
            continue

        for child in t.children:
            if child.is_space:
                continue
            ci = tok_idx.get(child.i)
            if ci is None:
                continue

            # Direktes Objekt (dep=oa): nur FVG-kompatible Nomina
            if child.dep_ in OBJ_DEPS and child.pos_ in ("NOUN", "PROPN"):
                if is_fvg_noun(child.lemma_, fvg_nouns):
                    if labels[vi] == "O":
                        labels[vi] = "B-VERB"
                    if labels[ci] == "O":
                        labels[ci] = "B-NOM"

            # PP-Modifier (dep=mo/cvc): Präp. + Nomen
            elif child.dep_ in PP_DEPS and child.pos_ == "ADP":
                if child.lemma_.lower() not in FVG_PREPS:
                    continue
                prep_ci = ci
                for pobj in child.children:
                    if pobj.is_space or pobj.dep_ != "nk":
                        continue
                    poi = tok_idx.get(pobj.i)
                    if poi is None:
                        continue
                    if pobj.pos_ in ("NOUN", "PROPN") and is_fvg_noun(pobj.lemma_, fvg_nouns):
                        if labels[vi] == "O":
                            labels[vi] = "B-VERB"
                        if labels[prep_ci] == "O":
                            labels[prep_ci] = "B-NOM"
                        if labels[poi] == "O":
                            labels[poi] = "I-NOM"


# ── Sätze streamen ─────────────────────────────────────────────────────────────

def iter_sentences(raw_file: Path, nlp) -> Iterator[tuple]:
    """Liefert (words, lemmas, spacy_sent) für jeden Satz."""
    def _texts():
        with open(raw_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)["text"]

    for doc in nlp.pipe(_texts(), batch_size=BATCH_SIZE, disable=["ner"]):
        for sent in doc.sents:
            tokens = [t for t in sent if not t.is_space]
            if MIN_SENT <= len(tokens) <= MAX_SENT:
                words  = [t.text        for t in tokens]
                lemmas = [t.lemma_.lower() for t in tokens]
                yield words, lemmas, sent


# ── Ausgabe ────────────────────────────────────────────────────────────────────

def write_split(samples, path):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  {len(samples):,} Sätze → {path.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEED)

    print("Lade spaCy-Modell …")
    nlp = spacy.load("de_core_news_lg", disable=["ner"])

    print("Lade FVG-Muster …")
    patterns, fvg_nouns = load_patterns(FVG_FILE, nlp)

    pos_samples, neg_samples = [], []

    print("Annotiere Sätze (Pass 1 + Pass 2) …")
    for words, lemmas, sent in tqdm(iter_sentences(RAW_FILE, nlp), desc="Sätze"):
        labels = ["O"] * len(words)

        pass1_list(words, lemmas, labels, patterns)
        pass2_syntax(sent, labels, fvg_nouns)

        sample = {"tokens": words, "labels": labels}
        if any(l != "O" for l in labels):
            pos_samples.append(sample)
        else:
            neg_samples.append(sample)

    neg_limit = min(len(neg_samples), len(pos_samples) * 2)
    random.shuffle(neg_samples)
    neg_samples = neg_samples[:neg_limit]

    all_samples = pos_samples + neg_samples
    random.shuffle(all_samples)

    n       = len(all_samples)
    n_train = int(n * 0.80)
    n_dev   = int(n * 0.10)
    train   = all_samples[:n_train]
    dev     = all_samples[n_train:n_train + n_dev]
    test    = all_samples[n_train + n_dev:]

    print(f"\nGesamt: {n:,} Sätze ({len(pos_samples):,} positiv, {len(neg_samples):,} negativ)")
    write_split(train, OUT_DIR / "annotated_train.jsonl")
    write_split(dev,   OUT_DIR / "annotated_dev.jsonl")
    write_split(test,  OUT_DIR / "annotated_test.jsonl")


if __name__ == "__main__":
    main()
