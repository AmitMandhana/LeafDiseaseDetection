"""
Small finetuning script to adapt a SentenceTransformers model to your `disease_data.json`.

What it does:
- Builds a simple dataset of (text, label) pairs by extracting key fields and generating short "farmer-style" paraphrases.
- Fine-tunes a pretrained sentence-transformer model (default: all-MiniLM-L6-v2) using a classification head.
- Saves the fine-tuned model to `model_finetuned/` which `app.py` will prefer.

Run with (venv active):
    .\.venv\Scripts\python.exe train_finetune.py

Notes:
- This is a minimal, CPU-friendly script intended as a starting point. For better results use more data, augmentation, and GPU training.
"""
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json
import random
import os

OUT_DIR = "model_finetuned"
BASE_MODEL = "all-MiniLM-L6-v2"  # small, fast baseline
EPOCHS = 3
BATCH_SIZE = 16

# helper to create short farmer-style queries from the disease entry
def synthesize_queries(entry, n=6):
    fields = []
    for key in ("leaf_symptoms", "fruit_effects", "disease_conditions", "plant_growth_effects"):
        if key in entry and entry[key]:
            fields.append(entry[key])
    text_pool = fields if fields else [json.dumps(entry)]
    queries = []
    templates = [
        "My plants have: {}",
        "Leaves are {}",
        "I see {} on leaves",
        "Fruit looks {}",
        "Weather is {} and plants look {}",
        "What is wrong if {}",
        "Plants showing: {}",
    ]
    for _ in range(n):
        src = random.choice(text_pool)
        tmpl = random.choice(templates)
        # take a short slice of the source to make queries concise
        piece = src.split('.')[0]
        # If template expects multiple placeholders, repeat the piece
        try:
            # count braces by using str.format_map safely
            placeholders = tmpl.count("{}")
            if placeholders <= 1:
                q = tmpl.format(piece)
            else:
                q = tmpl.format(*([piece] * placeholders))
        except Exception:
            q = tmpl.replace("{}", piece)
        # shorten overly long
        if len(q) > 240:
            q = q[:240]
        queries.append(q)
    return queries


def build_dataset(json_path="disease_data.json"):
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    texts = []
    labels = []
    label_map = {}
    for i, entry in enumerate(raw):
        label = entry.get('disease_name') or entry.get('name') or f'label_{i}'
        if label not in label_map:
            label_map[label] = len(label_map)
        lid = label_map[label]
        queries = synthesize_queries(entry, n=6)
        for q in queries:
            texts.append(q)
            labels.append(lid)
    # simple train/val split
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    split = int(len(combined) * 0.9)
    train = combined[:split]
    val = combined[split:]
    return train, val, label_map


def main():
    train, val, label_map = build_dataset()
    print(f"Built dataset: {len(train)} train, {len(val)} val, {len(label_map)} labels")

    model = SentenceTransformer(BASE_MODEL)

    # We'll fine-tune using a classification approach: mean pooling encoder + small MLP head.
    # sentence-transformers provides losses for such tasks; we'll use SoftmaxLoss.
    from sentence_transformers.losses import SoftmaxLoss
    num_labels = len(label_map)

    # SoftmaxLoss expects pair-shaped inputs (two texts) in this training loop.
    # To keep the dataset simple we duplicate the sentence as the second text.
    train_examples = [InputExample(texts=[t, t], label=l) for t, l in train]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=num_labels)

    # validation optional — we won't use a complicated evaluator here to keep it small

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=10,
        show_progress_bar=True
    )

    # save label map alongside model
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save(OUT_DIR)
    with open(os.path.join(OUT_DIR, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print('Saved fine-tuned model to', OUT_DIR)

if __name__ == '__main__':
    main()
