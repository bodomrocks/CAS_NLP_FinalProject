#!/usr/bin/env python3
"""
Schritt 15: Pair-Klassifikator v2 mit DS-augmentierten Daten trainieren.

Kombiniert Gold-Daten (pair_train.jsonl) mit Claude-annotierten DS-Paaren
(ds_claude_labels.jsonl). DS-Daten werden mit niedrigerem Gewicht trainiert,
da sie etwas verrauscht sind.

Gewichte:
  Gold positiv  : 1.0
  Gold negativ  : 1.0
  DS positiv    : 0.5  (Claude Precision ~0.90 → zuverlässig)
  DS negativ    : 0.3  (Claude Recall ~0.50 → konservativ, einige FVGs fehlen)

Ausgabe: models/fvg-pair-classifier-v2/
"""

import argparse
import json
import random
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

BASE_MODEL = "deepset/gbert-base"
MAX_LENGTH = 128

DATA_DIR  = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "fvg-pair-classifier-v2"

GOLD_TRAIN_JSONL = DATA_DIR / "pair_train.jsonl"
GOLD_TEST_JSONL  = DATA_DIR / "pair_test.jsonl"
DS_LABELS_JSONL  = DATA_DIR / "ds_claude_labels.jsonl"

SEED = 42

# Gewichte: Gold > DS
WEIGHT_GOLD_POS = 1.0
WEIGHT_GOLD_NEG = 1.0
WEIGHT_DS_POS   = 0.5
WEIGHT_DS_NEG   = 0.3

# Max. DS-Negativbeispiele (balanciert mit DS-Positiven)
DS_NEG_RATIO = 2.0   # DS-Neg = DS_NEG_RATIO × DS-Pos


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def make_record(sentence: str, verb_token: str, noun_token: str,
                praep: str, label: int, weight: float) -> dict:
    pattern = f"{praep} {noun_token}".strip() if praep else noun_token
    return {
        "sentence"  : sentence,
        "verb_token": verb_token,
        "noun_token": noun_token,
        "praep"     : praep,
        "text_b"    : f"{verb_token} {pattern}",
        "label"     : label,
        "weight"    : weight,
    }


def build_train_records(gold: list[dict], ds: list[dict]) -> list[dict]:
    records = []

    # Gold-Daten (unveränderter Gewichtung)
    for r in gold:
        w = WEIGHT_GOLD_POS if r["label"] == 1 else WEIGHT_GOLD_NEG
        records.append(make_record(
            r["sentence"], r["verb_token"], r["noun_token"],
            r.get("praep", ""), r["label"], w,
        ))

    # DS-Daten
    ds_pos = [r for r in ds if r["claude_label"] == "1"]
    ds_neg = [r for r in ds if r["claude_label"] == "0"]

    # DS-Negative auf DS_NEG_RATIO × DS-Positive begrenzen
    max_neg = int(len(ds_pos) * DS_NEG_RATIO)
    random.seed(SEED)
    random.shuffle(ds_neg)
    ds_neg = ds_neg[:max_neg]

    print(f"  DS positiv: {len(ds_pos)}  DS negativ (gesampelt): {len(ds_neg)}")

    for r in ds_pos:
        records.append(make_record(
            r["sentence"], r["verb_token"], r["noun_token"],
            r.get("praep", ""), 1, WEIGHT_DS_POS,
        ))
    for r in ds_neg:
        records.append(make_record(
            r["sentence"], r["verb_token"], r["noun_token"],
            r.get("praep", ""), 0, WEIGHT_DS_NEG,
        ))

    random.shuffle(records)
    return records


def make_dataset(records: list[dict], tokenizer) -> Dataset:
    enc = tokenizer(
        [r["sentence"] for r in records],
        [r["text_b"]   for r in records],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    enc["labels"]  = [r["label"]           for r in records]
    enc["weights"] = [r.get("weight", 1.0) for r in records]
    return Dataset.from_dict(enc)


def compute_metrics(pred: EvalPrediction) -> dict:
    preds  = np.argmax(pred.predictions, axis=-1)
    labels = pred.label_ids
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    prec   = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
    return {"precision": prec, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / len(labels),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weights", None)
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        loss    = loss_fn(logits, labels)
        if weights is not None:
            w = weights.to(loss.device).float()
            loss = (loss * w).mean()
        else:
            loss = loss.mean()
        return (loss, outputs) if return_outputs else loss


def main(push_to_hub: str | None = None):
    print(f"Lade Tokenizer: {BASE_MODEL} …")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Lade Daten …")
    gold_train = load_jsonl(GOLD_TRAIN_JSONL)
    gold_test  = load_jsonl(GOLD_TEST_JSONL)
    ds_labels  = load_jsonl(DS_LABELS_JSONL)

    ds_pos_count = sum(1 for r in ds_labels if r["claude_label"] == "1")
    ds_neg_count = sum(1 for r in ds_labels if r["claude_label"] == "0")
    print(f"  Gold Train: {len(gold_train)} "
          f"({sum(r['label'] for r in gold_train)} pos)")
    print(f"  DS Labels:  {len(ds_labels)} "
          f"({ds_pos_count} ja / {ds_neg_count} nein)")

    print("Erstelle Trainings-Records …")
    train_records = build_train_records(gold_train, ds_labels)
    test_records  = gold_test   # Testset bleibt unverändert

    n_pos = sum(r["label"] for r in train_records)
    n_neg = len(train_records) - n_pos
    print(f"  Train gesamt: {len(train_records)} ({n_pos} pos / {n_neg} neg)")

    train_ds = make_dataset(train_records, tokenizer)
    test_ds  = make_dataset(test_records,  tokenizer)

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
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
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
    )

    print("\nTraining startet …")
    trainer.train()

    print("\n" + "═" * 54)
    print("  EVALUATION auf Gold-Test-Split (n=244)")
    print("═" * 54)
    results = trainer.evaluate(test_ds)
    m = results
    print(f"  TP={m['eval_tp']}  FP={m['eval_fp']}  "
          f"FN={m['eval_fn']}  TN={m['eval_tn']}")
    print(f"  Precision : {m['eval_precision']:.3f}")
    print(f"  Recall    : {m['eval_recall']:.3f}")
    print(f"  F1        : {m['eval_f1']:.3f}")
    print(f"  Accuracy  : {m['eval_accuracy']:.3f}")

    # Vergleich mit v1
    print("\n  Vergleich:")
    print("  v1 (Gold only):  Prec=0.763  Rec=0.772  F1=0.768")
    print(f"  v2 (Gold + DS):  "
          f"Prec={m['eval_precision']:.3f}  "
          f"Rec={m['eval_recall']:.3f}  "
          f"F1={m['eval_f1']:.3f}")

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"\nModell gespeichert: {MODEL_DIR}")

    if push_to_hub:
        print(f"Pushe auf HuggingFace Hub: {push_to_hub} …")
        model.push_to_hub(push_to_hub)
        tokenizer.push_to_hub(push_to_hub)
        print("Fertig.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push_to_hub", default=None)
    args = ap.parse_args()
    main(args.push_to_hub)
