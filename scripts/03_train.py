#!/usr/bin/env python3
"""
Schritt 3: deepset/gbert-base auf FVG-Daten feintunen (Token Classification).

Voraussetzungen:
  - data/annotated_train.jsonl, annotated_dev.jsonl (aus Schritt 2)
  - CUDA-GPU empfohlen

Ausgabe: models/fvg-gbert-base/  (kompatibel mit HuggingFace Hub)
Laufzeit: ca. 1-3 Std. auf einer GPU (abhängig von Datengrösse)
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import classification_report, f1_score

BASE_MODEL       = "deepset/gbert-base"
DATA_DIR         = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR       = Path(__file__).resolve().parent.parent / "models" / "fvg-gbert-base"
GOLD_OVERSAMPLE  = 10   # Gold-Sätze N-fach wiederholen (Gewichtung)

LABEL_LIST = ["O", "B-VERB", "B-NOM", "I-NOM"]
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL   = {i: l for i, l in enumerate(LABEL_LIST)}


# ─── Daten laden ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def make_dataset(records: list[dict]) -> Dataset:
    return Dataset.from_dict({
        "tokens": [r["tokens"] for r in records],
        "labels": [r["labels"] for r in records],
    })


# ─── Tokenisierung mit Subword-Alignment ──────────────────────────────────────

def tokenize_and_align(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=512,
        is_split_into_words=True,
        padding=False,
    )

    aligned_labels = []
    for i, label_seq in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)      # [CLS], [SEP]
            elif word_id != prev_word_id:
                label_ids.append(LABEL2ID[label_seq[word_id]])
            else:
                # Folge-Subwörter: NOM-Kontinuation propagieren, VERB ignorieren
                lbl = label_seq[word_id]
                if lbl in ("B-NOM", "I-NOM"):
                    label_ids.append(LABEL2ID["I-NOM"])
                else:
                    label_ids.append(-100)   # B-VERB Subwörter ignorieren
            prev_word_id = word_id
        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


# ─── Evaluation ───────────────────────────────────────────────────────────────

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels  = []
    true_preds   = []

    for pred_seq, label_seq in zip(predictions, labels):
        sent_labels = []
        sent_preds  = []
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                sent_labels.append(ID2LABEL[label])
                sent_preds.append(ID2LABEL[pred])
        true_labels.append(sent_labels)
        true_preds.append(sent_preds)

    return {
        "f1":        f1_score(true_labels, true_preds),
        "report":    classification_report(true_labels, true_preds),
    }


# ─── Hauptprogramm ────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"GPU verfügbar: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Gerät: {torch.cuda.get_device_name(0)}")

    # Daten laden
    print("\nLade Trainingsdaten …")
    train_records = load_jsonl(DATA_DIR / "annotated_train.jsonl")
    dev_records   = load_jsonl(DATA_DIR / "annotated_dev.jsonl")
    print(f"  Train (distant): {len(train_records):,} | Dev: {len(dev_records):,}")

    gold_path = DATA_DIR / "gold_train.jsonl"
    if gold_path.exists():
        gold_records  = load_jsonl(gold_path)
        train_records = train_records + gold_records * GOLD_OVERSAMPLE
        print(f"  Gold-Labels: {len(gold_records)} × {GOLD_OVERSAMPLE} = "
              f"{len(gold_records) * GOLD_OVERSAMPLE} zusätzliche Sätze")
        print(f"  Train gesamt: {len(train_records):,}")
    else:
        print("  Kein gold_train.jsonl gefunden – nur Distant-Supervision-Daten.")

    raw_datasets = DatasetDict({
        "train": make_dataset(train_records),
        "validation": make_dataset(dev_records),
    })

    # Tokenizer
    print(f"\nLade Tokenizer: {BASE_MODEL} …")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    tokenized_datasets = raw_datasets.map(
        lambda ex: tokenize_and_align(ex, tokenizer),
        batched=True,
        remove_columns=["tokens", "labels"],
        desc="Tokenisierung",
    )

    # Modell
    print(f"\nLade Modell: {BASE_MODEL} …")
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Trainings-Argumente
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=200,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\nTraining startet …")
    trainer.train()

    print(f"\nModell gespeichert unter: {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Finales Evaluationsreport
    print("\n── Finales Evaluationsergebnis (Dev-Set) ──")
    metrics = trainer.evaluate()
    print(metrics.get("eval_report", ""))


if __name__ == "__main__":
    main()
