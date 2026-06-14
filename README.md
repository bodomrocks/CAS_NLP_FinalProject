# Automatic Detection of Funktionsverbgefüge in German Legal Texts

CAS Natural Language Processing — Final Project, June 2026

## Summary

This project builds a two-stage pipeline for automatically detecting **Funktionsverbgefüge (FVG)** — light verb constructions — in decisions of the Swiss Federal Administrative Court (BVGER). FVGs are multi-word expressions such as *Beschwerde erheben* ("to file a complaint") in which a semantically bleached function verb combines with a nominal component.

The pipeline combines **distant supervision** (auto-annotation from a 209-pattern reference list) with **two fine-tuned GBERT models**: an NER-based FVG Tagger and a binary Pair Classifier, both based on [`deepset/gbert-base`](https://huggingface.co/deepset/gbert-base). A syntactic post-filter based on spaCy dependency parsing further reduces false positives.

**[→ Try the live demo](https://huggingface.co/spaces/boj-per/fvg-demo)**

## Models

| Model | HF Hub | Task |
|---|---|---|
| FVG Tagger | [`boj-per/fvg-gbert-base`](https://huggingface.co/boj-per/fvg-gbert-base) | Token classification (BIO tagging) |
| Pair Classifier v2 | [`boj-per/fvg-pair-classifier-v2`](https://huggingface.co/boj-per/fvg-pair-classifier-v2) | Binary sequence classification |

## Results

| Component | Precision | Recall | F1 |
|---|---|---|---|
| Distant supervision (DS) quality | — | — | 0.42 |
| FVG Tagger (gold test set) | — | — | 0.35 |
| Pair Classifier v1 | — | — | 0.759 |
| Pair Classifier v2 (+ DS augmentation) | — | — | **0.777** |

The Pair Classifier v2 also uses 37,841 Claude-annotated DS pairs for augmentation (Claude Haiku 4.5; P=0.90, R=0.50, F1=0.64).

## Pipeline

```
Raw BVGER decisions (HTML)
        ↓ 01_extract_texts.py
Sentence corpus (~116K sentences)
        ↓ 02_annotate.py        ← Distant supervision (2-pass BIO annotation)
DS-annotated corpus
        ↓ 03_train.py           ← Fine-tune FVG Tagger (deepset/gbert-base)
FVG Tagger  →  04_push_to_hub.py
        ↓ 05_extract_eval_sentences.py + 06_annotate_gui.py
Gold standard (1,149 manually annotated sentences)
        ↓ 08_prepare_gold_train.py + 11_prepare_pair_data.py
Pair training data
        ↓ 14_claude_annotate_ds.py   ← Claude Haiku 4.5 DS augmentation
        ↓ 15_train_pair_classifier_v2.py
Pair Classifier v2  →  04_push_to_hub.py
        ↓ 16_corpus_analysis.py
Corpus-level FVG frequency statistics
```

## Project Structure

```
├── scripts/          Pipeline scripts (01–16, run in order)
├── app/
│   ├── demo.py               Gradio demo (two-model comparison)
│   └── fvg_syntax_filter.py  spaCy-based syntactic post-filter
├── data/
│   ├── liste_fvg.txt             209 FVG reference patterns (distant supervision input)
│   ├── verb_noun_fvg.csv         Verb-noun pair labels (eval input)
│   ├── praep_noun_verb.csv       Prepositional FVG patterns (eval input)
│   ├── gold_train/test.jsonl     Manually annotated gold standard
│   ├── pair_train/test.jsonl     Pair classifier training data
│   ├── eval_*.csv                Evaluation results
│   └── corpus_fvg_stats.json     Corpus-level FVG statistics
├── report/
│   ├── FVG_Detection_Report_Final.docx  Full project report (English)
│   └── Projektbericht_FVG-Erkennung.md  Project report (German)
└── requirements.txt
```

Large files not tracked in git: `data/raw_texts.jsonl`, `data/annotated_*.jsonl`, `data/corpus_fvg_results.jsonl`, `models/`, `bge_texte/`. See `.gitignore`.
