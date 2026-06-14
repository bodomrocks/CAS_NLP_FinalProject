#!/usr/bin/env python3
"""
FVG Demo — zwei Ansätze im Vergleich:
  Modell 1: FVG-Tagger (Token-Klassifikation, Distant Supervision)
  Modell 2: spaCy + Pair-Klassifikator v2

Starten:  python fvg_project/app/demo.py
"""

import re
from pathlib import Path
import torch
import spacy
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)

# ── Pfade ────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent.parent
NER_DIR  = BASE / "models" / "fvg-gbert-base"
PAIR_DIR = BASE / "models" / "fvg-pair-classifier-v2"

# ── Konstanten ────────────────────────────────────────────────────────────────
OBJ_DEPS  = {"oa", "oa2", "obj", "dobj"}
PP_DEPS   = {"mo", "cvc", "op", "pg", "obl"}
FVG_PREPS = {
    "in", "zu", "zur", "zum", "an", "auf", "unter", "über",
    "von", "vor", "nach", "bei", "mit", "gegen", "für", "ausser",
}
MAX_LEN  = 128
PUNCT_RE = re.compile(r"^[\W_]+|[\W_]+$")


def strip_punct(s: str) -> str:
    return PUNCT_RE.sub("", s)


# ── Modelle laden ─────────────────────────────────────────────────────────────
print("Lade FVG-Tagger …")
ner_tok   = AutoTokenizer.from_pretrained(str(NER_DIR))
ner_model = AutoModelForTokenClassification.from_pretrained(str(NER_DIR))
ner_model.eval()

print("Lade Pair-Klassifikator v2 …")
pair_tok   = AutoTokenizer.from_pretrained(str(PAIR_DIR))
pair_model = AutoModelForSequenceClassification.from_pretrained(str(PAIR_DIR))
pair_model.eval()

print("Lade spaCy (de_dep_news_trf) …")
nlp = spacy.load("de_dep_news_trf")
print("Bereit.\n")


# ── Pair-Klassifikator ────────────────────────────────────────────────────────

def _is_fvg(sentence: str, verb: str, noun: str, praep: str) -> bool:
    pattern = f"{praep} {noun}".strip() if praep else noun
    enc = pair_tok(sentence, f"{verb} {pattern}",
                   truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = pair_model(**enc).logits
    return int(logits.argmax(-1).item()) == 1


# ── Modell 1: FVG-Tagger ──────────────────────────────────────────────────────

def _ner_word_labels(text: str) -> list[tuple[str, str]]:
    """Gibt (word, label)-Paare auf Wortebene zurück (Leerzeichen-Tokenisierung)."""
    words = text.split()
    if not words:
        return []
    enc = ner_tok(words, is_split_into_words=True,
                  return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = ner_model(**enc).logits
    pred_ids = logits.argmax(-1)[0].tolist()
    id2label = ner_model.config.id2label
    word_ids = enc.word_ids(batch_index=0)
    word_label: dict[int, str] = {}
    for ti, wi in enumerate(word_ids):
        if wi is None or wi in word_label:
            continue
        word_label[wi] = id2label[pred_ids[ti]]
    return [(words[i], word_label.get(i, "O")) for i in range(len(words))]


def predict_tagger(text: str) -> list[tuple[str, str]]:
    return _ner_word_labels(text)


# ── Hilfsfunktion: NER-Paare extrahieren und mit v2 filtern ──────────────────

def _ner_confirmed_texts(text: str) -> tuple[set[str], set[str]]:
    """
    Gibt (confirmed_verb_texts, confirmed_nom_texts) zurück —
    Mengen bereinigter Tokentexte, die NER+v2 als FVG bestätigt hat.
    """
    word_labels = _ner_word_labels(text)

    # NOM-Spans sammeln
    nom_spans: list[tuple[int, list[int], list[str]]] = []
    i = 0
    while i < len(word_labels):
        word, label = word_labels[i]
        if label == "B-NOM":
            indices, tokens = [i], [word]
            j = i + 1
            while j < len(word_labels) and word_labels[j][1] == "I-NOM":
                indices.append(j)
                tokens.append(word_labels[j][0])
                j += 1
            nom_spans.append((i, indices, tokens))
            i = j
        else:
            i += 1

    verb_positions = [
        (i, strip_punct(w))
        for i, (w, l) in enumerate(word_labels)
        if l == "B-VERB" and strip_punct(w)
    ]

    used: set[int] = set()
    confirmed_verbs: set[str] = set()
    confirmed_noms:  set[str] = set()

    for verb_idx, verb_clean in verb_positions:
        best, best_dist = None, float("inf")
        for nom_start, nom_indices, nom_tokens in nom_spans:
            if nom_start in used:
                continue
            dist = abs(verb_idx - nom_start)
            if dist < best_dist:
                best_dist = dist
                best = (nom_start, nom_indices, nom_tokens)
        if best is None:
            continue
        nom_start, nom_indices, nom_tokens = best
        used.add(nom_start)

        clean = [strip_punct(t) for t in nom_tokens]
        first = clean[0]
        praep = first if first and first[0].islower() and len(first) <= 5 else ""
        noun  = clean[-1] if praep else clean[0]
        if not noun:
            continue

        if _is_fvg(text, verb_clean, noun, praep):
            confirmed_verbs.add(verb_clean.lower())
            for ni in nom_indices:
                confirmed_noms.add(strip_punct(word_labels[ni][0]).lower())

    return confirmed_verbs, confirmed_noms


# ── Modell 2: spaCy + v2 ─────────────────────────────────────────────────────

def _spacy_labels(text: str, doc=None) -> list[str]:
    """Interne Hilfsfunktion: gibt spaCy-Token-Labels zurück."""
    if doc is None:
        doc = nlp(text)
    labels = ["O"] * len(doc)
    for tok in doc:
        if tok.pos_ not in ("VERB", "AUX"):
            continue
        for child in tok.children:
            if child.dep_ in OBJ_DEPS:
                if _is_fvg(text, tok.text, child.text, ""):
                    labels[tok.i]   = "B-VERB"
                    labels[child.i] = "B-NOM"
            elif child.dep_ in PP_DEPS:
                if child.lemma_.lower() not in FVG_PREPS:
                    continue
                nk = [c for c in child.children if c.dep_ == "nk"]
                if not nk:
                    continue
                if _is_fvg(text, tok.text, nk[0].text, child.text):
                    labels[tok.i]   = "B-VERB"
                    labels[child.i] = "B-NOM"
                    labels[nk[0].i] = "I-NOM"
    return labels


def predict_spacy_v2(text: str) -> list[tuple[str, str]]:
    doc    = nlp(text)
    labels = _spacy_labels(text, doc)
    return [(t.text_with_ws.rstrip(), l) for t, l in zip(doc, labels)]



# ── HTML-Rendering ────────────────────────────────────────────────────────────

_VS = "background:#ffadad;border-radius:4px;padding:1px 5px;margin:1px;font-weight:bold"
_NS = "background:#a0c4ff;border-radius:4px;padding:1px 5px;margin:1px;font-weight:bold"


def to_html(pairs: list[tuple[str, str]]) -> str:
    parts = []
    for word, label in pairs:
        w = word.replace("&", "&amp;").replace("<", "&lt;")
        if label == "B-VERB":
            parts.append(f'<mark style="{_VS}">{w}</mark>')
        elif label in ("B-NOM", "I-NOM"):
            parts.append(f'<mark style="{_NS}">{w}</mark>')
        else:
            parts.append(w)
    body = " ".join(parts)
    return f'<p style="font-size:16px;line-height:2.4;font-family:serif">{body}</p>'


LEGEND = (
    '<div style="font-size:13px;margin-top:8px">'
    f'<mark style="{_VS}">Funktionsverb</mark>&nbsp;&nbsp;'
    f'<mark style="{_NS}">Nomen&nbsp;/&nbsp;Muster</mark>'
    "</div>"
)

EXAMPLES = [
    "Der Beschwerdeführer hat Beschwerde erhoben und einen Antrag gestellt.",
    "Die Behörde hat eine Verfügung erlassen und dem Betroffenen Kenntnis gegeben.",
    "Der Vertrag tritt in Kraft, sobald beide Parteien ihre Zustimmung erteilt haben.",
    "Das Gericht hat die Massnahme getroffen und den Richter in Ausstand versetzt.",
    (
        "Personen, die an Schlüsselpositionen eingesetzt werden, dürfen keinen Verrat üben "
        "und müssen die Gewähr bieten, das entgegengebrachte Vertrauen nicht zu missbrauchen."
    ),
]


def process(text: str):
    if not text.strip():
        msg = "<p><i>Bitte Text eingeben.</i></p>"
        return msg, msg
    h1 = to_html(predict_tagger(text))   + LEGEND
    h2 = to_html(predict_spacy_v2(text)) + LEGEND
    return h1, h2


# ── Gradio-Interface ──────────────────────────────────────────────────────────

with gr.Blocks(title="FVG Demo") as demo:
    gr.Markdown(
        "# Funktionsverbgefüge — Modellvergleich\n"
        "Gibt einen deutschen Satz (Rechtssprache) ein. "
        "Beide Modelle markieren erkannte FVG-Tokens."
    )
    inp = gr.Textbox(
        label="Eingabetext",
        placeholder="z.B. Der Beschwerdeführer hat Beschwerde erhoben und einen Antrag gestellt.",
        lines=3,
    )
    btn = gr.Button("Analysieren", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Modell 1: FVG-Tagger")
            gr.Markdown("*Token-Klassifikation, Distant Supervision, F1=0.548*")
            out1 = gr.HTML()
        with gr.Column():
            gr.Markdown("### Modell 2: spaCy + Pair-Klassifikator")
            gr.Markdown("*Syntaktische Kandidatenfindung + v2, F1=0.777*")
            out2 = gr.HTML()

    btn.click(fn=process, inputs=inp, outputs=[out1, out2])
    inp.submit(fn=process, inputs=inp, outputs=[out1, out2])
    gr.Examples(examples=EXAMPLES, inputs=inp, label="Beispielsätze")

if __name__ == "__main__":
    demo.launch(share=False)
