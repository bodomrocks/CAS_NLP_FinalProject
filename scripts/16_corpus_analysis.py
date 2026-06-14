#!/usr/bin/env python3
"""
Skript 16: Large-Scale FVG-Analyse auf dem BGE-Korpus.

Input:  ../../bge_texte/*.html  (40 930 HTML-Dateien, direkt)
Output: data/corpus_fvg_results.jsonl   -- pro Satz mit FVG: Quelldokument + gefundene FVGs
        data/corpus_fvg_stats.json      -- Gesamtstatistik + Top-50-Listen

Laufzeit (de_core_news_lg, GPU RTX 4070): ca. 8-15 h
Laufzeit (de_core_news_lg, CPU):          ca. 40-100 h

Das Script kann jederzeit unterbrochen und fortgesetzt werden (Checkpoint alle 100 Dok.).

Aufruf:
    python scripts/16_corpus_analysis.py               # Beide Modelle
    python scripts/16_corpus_analysis.py --no-tagger   # Nur spaCy+v2 (schneller)
    python scripts/16_corpus_analysis.py --limit 100   # Test mit 100 Dokumenten
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
import spacy
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

# ── Pfade ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parent.parent
BGE_DIR   = BASE.parent / "bge_texte"
OUT_JSONL = BASE / "data" / "corpus_fvg_results.jsonl"
OUT_STATS = BASE / "data" / "corpus_fvg_stats.json"
CKPT_FILE = BASE / "data" / "corpus_fvg_checkpoint.json"
NER_DIR   = BASE / "models" / "fvg-gbert-base"
PAIR_DIR  = BASE / "models" / "fvg-pair-classifier-v2"

# ── Konstanten ────────────────────────────────────────────────────────────────
OBJ_DEPS   = {"oa", "oa2", "obj", "dobj"}
PP_DEPS    = {"mo", "cvc", "op", "pg", "obl"}
FVG_PREPS  = {
    "in", "zu", "zur", "zum", "an", "auf", "unter", "über",
    "von", "vor", "nach", "bei", "mit", "gegen", "für", "ausser",
}
MAX_LEN        = 128
MAX_SENT_CHARS = 600
MAX_SEG_CHARS  = 2000
PAIR_BATCH     = 64   # Kandidaten pro Batch im Pair-Klassifikator (GPU: höherer Wert sinnvoll)
PUNCT_RE       = re.compile(r"^[\W_]+|[\W_]+$")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def strip_punct(s: str) -> str:
    return PUNCT_RE.sub("", s)


# ── HTML-Bereinigung ──────────────────────────────────────────────────────────
_HTML_TAG_RE = re.compile(r"<[^>]+>")

def extract_text_from_html(path: Path) -> str | None:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    text = _HTML_TAG_RE.sub(" ", raw)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text if len(text) >= 200 else None


# ── Modelle laden ─────────────────────────────────────────────────────────────

def load_models(spacy_model: str, use_tagger: bool):
    print(f"Gerät: {DEVICE}", flush=True)
    print(f"Lade spaCy ({spacy_model}) …", flush=True)
    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        fallback = "de_core_news_lg" if "trf" in spacy_model else "de_dep_news_trf"
        print(f"  Nicht gefunden — Fallback auf {fallback}", flush=True)
        nlp = spacy.load(fallback)
    nlp.max_length = 2_000_000

    print("Lade Pair-Klassifikator v2 …", flush=True)
    pair_tok = AutoTokenizer.from_pretrained(str(PAIR_DIR))
    pair_model = AutoModelForSequenceClassification.from_pretrained(str(PAIR_DIR))
    pair_model.to(DEVICE).eval()

    ner_tok = ner_model = None
    if use_tagger:
        print("Lade FVG-Tagger …", flush=True)
        ner_tok = AutoTokenizer.from_pretrained(str(NER_DIR))
        ner_model = AutoModelForTokenClassification.from_pretrained(str(NER_DIR))
        ner_model.to(DEVICE).eval()

    print("Bereit.\n", flush=True)
    return nlp, pair_tok, pair_model, ner_tok, ner_model


# ── Pair-Klassifikator (Batch) ────────────────────────────────────────────────

def classify_candidates_batch(pair_tok, pair_model, candidates: list[dict]) -> list[bool]:
    """
    Batch-Inferenz für den Pair-Klassifikator (CPU oder GPU).
    candidates: Liste von {sent, verb, noun, praep, verb_lemma, pattern_lemma}
    """
    if not candidates:
        return []
    results: list[bool] = []
    for i in range(0, len(candidates), PAIR_BATCH):
        batch = candidates[i : i + PAIR_BATCH]
        texts_a = [c["sent"] for c in batch]
        texts_b = [
            f"{c['verb']} {(c['praep'] + ' ' + c['noun']).strip()}"
            if c["praep"] else f"{c['verb']} {c['noun']}"
            for c in batch
        ]
        enc = pair_tok(
            texts_a, texts_b,
            truncation=True, max_length=MAX_LEN,
            padding=True, return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = pair_model(**enc).logits
        results.extend((logits.argmax(-1) == 1).tolist())
    return results


# ── Modell 1: FVG-Tagger ──────────────────────────────────────────────────────

def predict_tagger(ner_tok, ner_model, text: str) -> list[tuple[str, str]]:
    words = text.split()
    if not words:
        return []
    enc = ner_tok(words, is_split_into_words=True,
                  return_tensors="pt", truncation=True, max_length=512)
    word_ids = enc.word_ids(batch_index=0)   # vor Device-Transfer abrufen
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = ner_model(**enc).logits
    pred_ids = logits.argmax(-1)[0].tolist()
    id2label = ner_model.config.id2label
    word_label: dict[int, str] = {}
    for ti, wi in enumerate(word_ids):
        if wi is None or wi in word_label:
            continue
        word_label[wi] = id2label[pred_ids[ti]]
    return [(words[i], word_label.get(i, "O")) for i in range(len(words))]


def extract_tagger_fvgs(word_labels: list[tuple[str, str]]) -> list[dict]:
    verb_positions = [
        (i, strip_punct(w).lower())
        for i, (w, l) in enumerate(word_labels)
        if l == "B-VERB" and strip_punct(w)
    ]
    nom_spans: list[tuple[int, list[str]]] = []
    i = 0
    while i < len(word_labels):
        w, l = word_labels[i]
        if l == "B-NOM":
            tokens = [strip_punct(w)]
            j = i + 1
            while j < len(word_labels) and word_labels[j][1] == "I-NOM":
                tokens.append(strip_punct(word_labels[j][0]))
                j += 1
            nom_spans.append((i, [t for t in tokens if t]))
            i = j
        else:
            i += 1
    if not verb_positions or not nom_spans:
        return []
    fvgs, used = [], set()
    for verb_idx, verb_clean in verb_positions:
        best = min(nom_spans, key=lambda x: abs(x[0] - verb_idx), default=None)
        if best is None or best[0] in used:
            continue
        used.add(best[0])
        nom_text = " ".join(best[1]).lower()
        if nom_text:
            fvgs.append({"verb": verb_clean, "pattern": nom_text})
    return fvgs


# ── Modell 2: spaCy-Kandidaten sammeln ────────────────────────────────────────

def collect_spacy_candidates(sent) -> list[dict]:
    """Gibt Kandidaten zurück ohne Pair-Klassifikation."""
    sent_text = sent.text
    candidates = []
    for tok in sent:
        if tok.pos_ not in ("VERB", "AUX"):
            continue
        for child in tok.children:
            if child.dep_ in OBJ_DEPS:
                candidates.append({
                    "sent":          sent_text,
                    "verb":          tok.text,
                    "noun":          child.text,
                    "praep":         "",
                    "verb_lemma":    tok.lemma_.lower(),
                    "pattern_lemma": child.lemma_.lower(),
                })
            elif child.dep_ in PP_DEPS:
                praep = child.lemma_.lower()
                if praep not in FVG_PREPS:
                    continue
                nk = [c for c in child.children if c.dep_ == "nk"]
                if not nk:
                    continue
                candidates.append({
                    "sent":          sent_text,
                    "verb":          tok.text,
                    "noun":          nk[0].text,
                    "praep":         child.text,
                    "verb_lemma":    tok.lemma_.lower(),
                    "pattern_lemma": f"{praep} {nk[0].lemma_.lower()}",
                })
    return candidates


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(docs_done: int, stats: dict,
                    counter_tagger: Counter, counter_spacy: Counter):
    CKPT_FILE.write_text(json.dumps({
        "docs_done":      docs_done,
        "stats":          stats,
        "counter_tagger": dict(counter_tagger.most_common(2000)),
        "counter_spacy":  dict(counter_spacy.most_common(2000)),
    }, ensure_ascii=False), encoding="utf-8")


def load_checkpoint() -> tuple[int, dict, Counter, Counter]:
    empty = {"docs": 0, "sents": 0, "tagger_with_fvg": 0, "spacy_with_fvg": 0}
    if not CKPT_FILE.exists():
        return 0, empty, Counter(), Counter()
    ckpt = json.loads(CKPT_FILE.read_text(encoding="utf-8"))
    return (
        ckpt["docs_done"],
        ckpt.get("stats", empty),
        Counter(ckpt.get("counter_tagger", {})),
        Counter(ckpt.get("counter_spacy", {})),
    )


# ── Hauptprogramm ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Large-Scale FVG-Analyse auf BGE-Korpus")
    ap.add_argument("--limit", type=int, default=0,
                    help="Maximale Anzahl Dokumente (0 = alle)")
    ap.add_argument("--spacy-model", choices=["trf", "lg"], default="lg",
                    help="spaCy-Modell: lg (Standard, schnell) oder trf (genauer)")
    ap.add_argument("--no-tagger", action="store_true",
                    help="FVG-Tagger (Modell 1) nicht ausführen")
    ap.add_argument("--checkpoint-every", type=int, default=100,
                    help="Checkpoint-Intervall in Dokumenten (Standard: 100)")
    ap.add_argument("--reset", action="store_true",
                    help="Checkpoint verwerfen und neu starten")
    args = ap.parse_args()

    spacy_name = "de_dep_news_trf" if args.spacy_model == "trf" else "de_core_news_lg"
    use_tagger = not args.no_tagger

    nlp, pair_tok, pair_model, ner_tok, ner_model = load_models(spacy_name, use_tagger)

    if args.reset and CKPT_FILE.exists():
        CKPT_FILE.unlink()
    start_doc, stats, counter_tagger, counter_spacy = load_checkpoint()
    if start_doc > 0:
        print(f"Setze fort ab Dokument {start_doc:,} …", flush=True)

    all_files = sorted(BGE_DIR.glob("*.html"))
    if args.limit:
        all_files = all_files[: args.limit]
    total_docs = len(all_files)
    print(f"BGE-Dateien gesamt: {total_docs:,}", flush=True)

    file_mode = "a" if start_doc > 0 else "w"
    with open(OUT_JSONL, file_mode, encoding="utf-8") as fout:
        for doc_idx, html_path in enumerate(
            tqdm(all_files[start_doc:], desc="Dokumente", unit="dok"),
            start=start_doc,
        ):
            text = extract_text_from_html(html_path)
            if not text:
                stats["docs"] = doc_idx + 1
                continue

            segments = re.split(r"\n{2,}", text)

            # Alle Kandidaten des Dokuments für Batch-Klassifikation sammeln
            # Format: (sent_idx_key, candidate_dict)
            # sent_idx_key → (sent_text, tagger_fvgs, spacy_candidate_list)
            sent_records: dict[str, dict] = {}  # key = sent_text (eindeutig genug)

            for seg in segments:
                seg = seg.strip()
                if len(seg) < 20:
                    continue
                if len(seg) > MAX_SEG_CHARS:
                    seg = seg[:MAX_SEG_CHARS]
                try:
                    doc = nlp(seg)
                except Exception:
                    continue

                for sent in doc.sents:
                    sent_text = sent.text.strip()
                    if len(sent_text) < 20 or len(sent_text) > MAX_SENT_CHARS:
                        continue
                    stats["sents"] += 1

                    # Tagger (bleibt sequenziell, da word_ids-Mapping je Satz)
                    fvgs_tagger: list[dict] = []
                    if use_tagger:
                        wl = predict_tagger(ner_tok, ner_model, sent_text)
                        fvgs_tagger = extract_tagger_fvgs(wl)

                    # spaCy-Kandidaten sammeln (noch keine Klassifikation)
                    candidates = collect_spacy_candidates(sent)

                    # Satz merken (bei Duplikat im selben Dokument: überschreiben)
                    sent_records[sent_text] = {
                        "tagger":     fvgs_tagger,
                        "candidates": candidates,
                    }

            # Alle gesammelten Kandidaten dieses Dokuments in einem Batch klassifizieren
            all_candidates: list[dict] = []
            sent_order: list[str] = []
            cand_offsets: list[int] = [0]  # Startindex pro Satz

            for sent_text, rec in sent_records.items():
                sent_order.append(sent_text)
                all_candidates.extend(rec["candidates"])
                cand_offsets.append(len(all_candidates))

            classifications = classify_candidates_batch(pair_tok, pair_model, all_candidates)

            # Ergebnisse je Satz zusammenführen und schreiben
            for i, sent_text in enumerate(sent_order):
                rec = sent_records[sent_text]
                fvgs_tagger = rec["tagger"]

                start = cand_offsets[i]
                end   = cand_offsets[i + 1]
                fvgs_spacy = [
                    {"verb": c["verb_lemma"], "pattern": c["pattern_lemma"]}
                    for c, is_fvg in zip(all_candidates[start:end], classifications[start:end])
                    if is_fvg
                ]

                if fvgs_tagger:
                    stats["tagger_with_fvg"] += 1
                    for fvg in fvgs_tagger:
                        counter_tagger[f"{fvg['verb']} {fvg['pattern']}"] += 1

                if fvgs_spacy:
                    stats["spacy_with_fvg"] += 1
                    for fvg in fvgs_spacy:
                        counter_spacy[f"{fvg['verb']} {fvg['pattern']}"] += 1

                if fvgs_tagger or fvgs_spacy:
                    fout.write(json.dumps({
                        "doc":      html_path.name,
                        "text":     sent_text,
                        "tagger":   fvgs_tagger,
                        "spacy_v2": fvgs_spacy,
                    }, ensure_ascii=False) + "\n")

            stats["docs"] = doc_idx + 1

            if (doc_idx + 1) % args.checkpoint_every == 0:
                save_checkpoint(doc_idx + 1, stats, counter_tagger, counter_spacy)
                tqdm.write(
                    f"  [Checkpoint] Dok {doc_idx+1:,}/{total_docs:,} | "
                    f"Sätze: {stats['sents']:,} | "
                    f"FVG Tagger/spaCy: {stats['tagger_with_fvg']:,}/{stats['spacy_with_fvg']:,}"
                )

    # Checkpoint löschen, finale Statistik schreiben
    if CKPT_FILE.exists():
        CKPT_FILE.unlink()

    sents = max(1, stats["sents"])
    output_stats = {
        **stats,
        "tagger_fvg_rate": round(stats["tagger_with_fvg"] / sents, 4),
        "spacy_fvg_rate":  round(stats["spacy_with_fvg"]  / sents, 4),
        "top50_tagger":    counter_tagger.most_common(50),
        "top50_spacy_v2":  counter_spacy.most_common(50),
    }
    OUT_STATS.write_text(
        json.dumps(output_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"Dokumente:                {stats['docs']:>12,}")
    print(f"Sätze (analysiert):       {stats['sents']:>12,}")
    if use_tagger:
        pct = 100 * stats["tagger_with_fvg"] / sents
        print(f"Sätze mit FVG (Tagger):   {stats['tagger_with_fvg']:>12,}  ({pct:.1f} %)")
    pct = 100 * stats["spacy_with_fvg"] / sents
    print(f"Sätze mit FVG (spaCy+v2): {stats['spacy_with_fvg']:>12,}  ({pct:.1f} %)")

    print(f"\nTop-20 FVGs (spaCy+v2, lemmatisiert):")
    for rank, (fvg, n) in enumerate(counter_spacy.most_common(20), 1):
        print(f"  {rank:2d}. {n:7,} ×  {fvg}")
    if use_tagger:
        print(f"\nTop-20 FVGs (Tagger):")
        for rank, (fvg, n) in enumerate(counter_tagger.most_common(20), 1):
            print(f"  {rank:2d}. {n:7,} ×  {fvg}")

    print(f"\nErgebnisse: {OUT_JSONL}")
    print(f"Statistik:  {OUT_STATS}")


if __name__ == "__main__":
    main()
