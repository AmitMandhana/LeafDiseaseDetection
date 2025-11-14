# LeafDiseaseDetection

## Project overview

LeafDiseaseDetection is a small, self-contained assistant that helps identify plant (tomato) diseases by matching user-written symptom descriptions to a curated disease corpus. The project uses text embeddings (SentenceTransformers) rather than image classification: disease descriptions are embedded into a vector space and user queries are matched via cosine similarity.

This repository contains scripts to build a synthetic symptom dataset from `disease_data.json`, fine-tune a sentence-transformer, evaluate models, and run a Streamlit demo app for interactive prediction and disambiguation.

Highlights
- Lightweight baseline: uses `all-MiniLM-L6-v2` (a small, fast SentenceTransformers model).
- Fine-tuning approach: mean-pooling encoder + small softmax MLP head (SoftmaxLoss) to adapt embeddings to the disease retrieval task.
- Inference: encode user query, compute cosine similarity against pre-computed disease description embeddings, return the top match.

---

## Quick demo

Requirements (from `requirements.txt`) and run (Windows PowerShell):

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the Streamlit UI that appears (usually http://localhost:8501) and enter symptom text to test the assistant.

---

## Repo structure (important files)

- `app.py` — Streamlit application. Loads the model and the disease corpus, creates embeddings, and exposes the interactive UI (predict + disambiguation flow).
- `train_finetune.py` — Small finetuning script that:
  - Synthesizes short ‘‘farmer-style’’ queries from fields in `disease_data.json`.
  - Builds a (text, label) dataset.
  - Fine-tunes a SentenceTransformer (`all-MiniLM-L6-v2` by default) using `SoftmaxLoss` and saves to `model_finetuned/`.
- `evaluate_models.py` — Simple script to compute top-k retrieval accuracy on the synthetic validation split for the local, base, and finetuned models.
- `compare_models.py` — (helper scripts; optional) used to compare embeddings or visualize differences.
- `diag_model.py` — quick inspection script that loads the local `model/` to sanity-check the SentenceTransformer object.
- `disease_data.json` — the corpus of disease entries. Each entry contains fields like `name`, `leaf_symptoms`, `description`, etc., used to synthesize queries and to create the retrieval corpus.
- `model/` — the original model folder (pretrained or downloaded checkpoint).
- `model_finetuned/` — the fine-tuned model saved by `train_finetune.py`. Also contains `label_map.json` produced during training.

---

## Data pipeline (how data is created and used)

1. Input corpus: `disease_data.json` — each entry includes disease name, symptoms, and optional human-readable descriptions.
2. `train_finetune.py` builds synthetic queries by sampling and templating fields like `leaf_symptoms` and `fruit_effects` (function `synthesize_queries`).
3. Train/Val split: a simple random split (default 90% train / 10% val).
4. Fine-tune SentenceTransformers using SoftmaxLoss, which trains an MLP classification head on top of sentence embeddings (encoder + mean pooling).
5. Save model and `label_map.json` in `model_finetuned/`.

At inference time (in `app.py`):
1. Load either `model_finetuned/` (preferred) or `model/`.
2. Build corpus descriptions and pre-compute their embeddings (stored in memory for the Streamlit session).
3. For each user query, encode the query to an embedding and compute cosine similarity versus the corpus embeddings.
4. Return the top match (and similarity score) to the user. The app also provides a disambiguation flow to ask targeted yes/no questions when two diseases are similar.

---

## Model architecture and training details

This section explains the model components used by the code and how they interact. The diagram file `assets/model_structure.svg` in this repo illustrates the same flow; include it in your PPT slides for a clear visual.

Contract (inputs / outputs)
- Input: short textual descriptions of symptoms (free-form text assembled from UI fields).
- Output: ranked disease matches (top-k) with a confidence score (cosine similarity).
- Success criteria: high top-1 / top-3 retrieval accuracy on synthetic validation queries extracted from `disease_data.json`.

Model components (as used in the code):
- Base encoder: `all-MiniLM-L6-v2` (a small Transformer trained for sentence embeddings). This model outputs token embeddings.
- Pooling: mean pooling over the token embeddings to produce a fixed-size sentence embedding (the sentence-transformers pooling layer).
- Fine-tuning head: SoftmaxLoss adds a small fully-connected classification head (dimension = number of labels) on top of the pooled embedding during training. This head helps cluster the disease descriptions so retrieval via cosine similarity works better.

Training hyperparameters (as found in `train_finetune.py`):
- BASE_MODEL: `all-MiniLM-L6-v2`
- EPOCHS: 3
- BATCH_SIZE: 16
- Warmup steps: 10
- Loss: `SoftmaxLoss` (classification/softmax over labels using the learned MLP head)

Why this architecture?
- Speed: MiniLM is fast and small—reasonable for CPU/GPU and for quick demos.
- Retrieval-friendly: Fine-tuning with SoftmaxLoss encourages separation of disease classes in embedding space, which helps nearest-neighbor retrieval.

Limitations and edge-cases
- The approach treats the problem as text retrieval, not image classification. It relies on textual descriptions in `disease_data.json` and user-provided text.
- Quality depends heavily on the coverage and quality of the disease descriptions.
- If diseases have very similar text descriptions, the system may confuse them; the app provides a disambiguation flow for that case.

---

## Diagram (model + inference) 

The diagram file `assets/model_structure.svg` is included and shows both the training and inference paths: data synthesis -> encoder + pooling -> softmax head (training), and query encoding -> cosine similarity -> top-k retrieval (inference).

Include this SVG in your slide deck (PowerPoint supports SVG import on modern Office). If your PPT tool doesn't support SVG, export to PNG at a high DPI.

---

## How to reproduce training & evaluation

1. Install requirements and create a virtual environment (optional but recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

2. Fine-tune the model (will save to `model_finetuned/`):

```powershell
python train_finetune.py
```

3. Evaluate models (computes top-k accuracy on synthetic val set):

```powershell
python evaluate_models.py
```

4. Run the Streamlit demo:

```powershell
streamlit run app.py
```

Notes: training is intentionally small (3 epochs, small batches). For production or better accuracy, use more data, longer training, and a GPU environment.

---

## Files you can show in the PPT (suggested order)

1. `disease_data.json` — show a single disease entry to illustrate fields used.
2. `train_finetune.py` — a short code snippet (synthesize_queries and model.fit) to explain dataset creation and training.
3. `model_finetuned/label_map.json` — explain label ids and mapping.
4. `app.py` — inference flow: encode query, compute similarity, show match and confidence.
5. `assets/model_structure.svg` — the visual architecture diagram.

---

## Slide-by-slide suggestions (3–6 minute walkthrough)

Slide 1 — Title & Goal
- Title: Leaf Disease Detection — Text-based Retrieval Assistant
- Speaker notes: "Goal: help farmers/experts identify crop diseases from short symptom descriptions."

Slide 2 — Data & Corpus
- Show single `disease_data.json` entry and explain fields used (symptoms, description).
- Speaker notes: "We synthesize queries from symptom fields so the model learns phrasing variability."

Slide 3 — Model Architecture (use `assets/model_structure.svg`)
- Explain base encoder (MiniLM), pooling, and softmax head used for fine-tuning.
- Speaker notes: "We fine-tune to cluster disease descriptions in embedding space."

Slide 4 — Training & Evaluation
- Show the small code flow and hyperparameters (epochs=3, batch=16), then present top-1/top-3 accuracy numbers if you ran evaluation.

Slide 5 — Live Demo / App Flow
- Show the Streamlit UI screenshot and explain the prediction + disambiguation interaction.

Slide 6 — Limitations & Next Steps
- Mention reliance on text, need for richer corpus, multi-modal extension (images + text), and improved training.

---

## Notes, assumptions and next steps

Assumptions made while documenting:
- The project uses a text-based retrieval approach (the code confirms SentenceTransformers use). If you intended an image-based model, the current repository is text-first and would need a different data pipeline.

Potential next steps (small, low-risk enhancements):
- Add a saved `corpus_embeddings.pt` cache to avoid recomputing embeddings at app start.
- Add optional image upload + vision model pipeline (for multimodal inference).
- Add a small Jupyter notebook showing example predictions and evaluation plots.

---

## License

Include your preferred license here (not currently set in repo). If none is provided, add a short license (MIT) or ask the project owner which to use.

---

If you want, I can also:
- produce a PowerPoint (.pptx) with these slides and embed `assets/model_structure.svg` for you, or
- export a high-resolution PNG of the diagram for older PowerPoint versions.

Tell me which option you prefer and I will generate it next.
