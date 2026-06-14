#!/usr/bin/env python3
"""
Schritt 12: Paarklassifikator für FVG trainieren.

Feintuning von deepset/gbert-base als binärer Sequenzklassifikator:
  Input : [CLS] Satz [SEP] verb_token pattern [SEP]
  Output: 0 (kein FVG) | 1 (FVG)

Trainings- und Testdaten: data/pair_train.jsonl, data/pair_test.jsonl
Ausgabe: models/fvg-pair-classifier/

Verwendung:
  python 12_train_pair_classifier.py
  python 12_train_pair_classifier.py --push_to_hub boj-per/fvg-pair-classifier
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from sklearn.metrics import classification_report, confusion_matrix

BASE_MODEL = "deepset/gbert-base"
MAX_LENGTH = 128

DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR  = Path(__file__).resolve().parent.parent / "models" / "fvg-pair-classifier"

TRAIN_JSONL = DATA_DIR / "pair_train.jsonl"
TEST_JSONL  = DATA_DIR / "pair_test.jsonl"


# ── Daten laden ────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Tokenisierung ──────────────────────────────────────────────────────────────

def make_dataset(records: list[dict], tokenizer) -> Dataset:
    texts_a = [r["sentence"] for r in records]
    texts_b = [r["text_b"]   for r in records]
    labels  = [r["label"]    for r in records]

    enc = tokenizer(
        texts_a,
        texts_b,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    enc["labels"] = labels
    return Dataset.from_dict(enc)


# ── Metriken ───────────────────────────────────────────────────────────────────

def compute_metrics(pred: EvalPrediction) -> dict:
    logits = pred.predictions
    labels = pred.label_ids
    preds  = np.argmax(logits, axis=-1)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    prec   = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
    acc    = (tp + tn) / len(labels)

    return {"precision": prec, "recall": recall, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ── Gewichteter Verlust (Klassenimbalance) ─────────────────────────────────────

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ── Hauptprogramm ──────────────────────────────────────────────────────────────

def main(push_to_hub: str | None = None):
    print(f"Lade Tokenizer: {BASE_MODEL} …")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Lade Daten …")
    train_records = load_jsonl(TRAIN_JSONL)
    test_records  = load_jsonl(TEST_JSONL)
    print(f"  Train: {len(train_records)} ({sum(r['label'] for r in train_records)} pos)")
    print(f"  Test:  {len(test_records)}  ({sum(r['label'] for r in test_records)} pos)")

    train_ds = make_dataset(train_records, tokenizer)
    test_ds  = make_dataset(test_records,  tokenizer)

    # Klassengewichte: Ausgleich für Positiv/Negativ-Ungleichgewicht
    n_pos = sum(r["label"] for r in train_records)
    n_neg = len(train_records) - n_pos
    w_pos = n_neg / (n_pos + n_neg)   # Positiv: höheres Gewicht wenn seltener
    w_neg = n_pos / (n_pos + n_neg)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float)
    print(f"  Klassengewichte: neg={w_neg:.3f}, pos={w_pos:.3f}")

    print(f"\nLade Modell: {BASE_MODEL} …")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "KEIN_FVG", 1: "FVG"},
        label2id={"KEIN_FVG": 0, "FVG": 1},
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=2,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print("\nTraining startet …")
    trainer.train()

    print("\n" + "═" * 54)
    print("  EVALUATION auf Test-Split")
    print("═" * 54)
    results = trainer.evaluate(test_ds)
    m = results
    print(f"  TP={m['eval_tp']}  FP={m['eval_fp']}  FN={m['eval_fn']}  TN={m['eval_tn']}")
    print(f"  Precision : {m['eval_precision']:.3f}")
    print(f"  Recall    : {m['eval_recall']:.3f}")
    print(f"  F1        : {m['eval_f1']:.3f}")
    print(f"  Accuracy  : {m['eval_accuracy']:.3f}")

    # Fehleranalyse
    pred_out = trainer.predict(test_ds)
    preds    = np.argmax(pred_out.predictions, axis=-1)
    labels   = np.array([r["label"] for r in test_records])

    fn_idx = [i for i, (p, l) in enumerate(zip(preds, labels)) if p == 0 and l == 1]
    fp_idx = [i for i, (p, l) in enumerate(zip(preds, labels)) if p == 1 and l == 0]

    print(f"\n  False Negatives ({len(fn_idx)}):")
    for i in fn_idx[:6]:
        r = test_records[i]
        print(f"    {r['verb_lemma']} + {r['noun_lemma']}: {r['sentence'][:80]}")

    print(f"\n  False Positives ({len(fp_idx)}):")
    for i in fp_idx[:6]:
        r = test_records[i]
        print(f"    {r['verb_lemma']} + {r['noun_lemma']}: {r['sentence'][:80]}")

    # Modell speichern
    save_path = str(MODEL_DIR)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nModell gespeichert: {save_path}")

    if push_to_hub:
        print(f"Pushe auf HuggingFace Hub: {push_to_hub} …")
        model.push_to_hub(push_to_hub)
        tokenizer.push_to_hub(push_to_hub)
        print("Fertig.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push_to_hub", default=None,
                    help="HF-Hub-Repo, z.B. boj-per/fvg-pair-classifier")
    args = ap.parse_args()
    main(args.push_to_hub)
