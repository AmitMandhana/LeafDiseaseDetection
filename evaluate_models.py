from sentence_transformers import SentenceTransformer, util
import json, random
import torch
from sklearn.metrics import hamming_loss, precision_score, recall_score, f1_score

# recreate the same dataset generation used in train_finetune.py
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
        piece = src.split('.')[0]
        try:
            placeholders = tmpl.count("{}")
            if placeholders <= 1:
                q = tmpl.format(piece)
            else:
                q = tmpl.format(*([piece] * placeholders))
        except Exception:
            q = tmpl.replace("{}", piece)
        if len(q) > 240:
            q = q[:240]
        queries.append(q)
    return queries


def build_dataset(json_path='disease_data.json'):
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
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    split = int(len(combined) * 0.9)
    train = combined[:split]
    val = combined[split:]
    return train, val, label_map, raw


def evaluate_model(model_name, model, val_set, corpus):
    corpus_embeddings = model.encode([c['description'] for c in corpus], convert_to_tensor=True)
    y_true = []
    y_pred = []

    for text, true_labels in val_set:
        q_emb = model.encode(text, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(q_emb, corpus_embeddings)[0]
        predicted_labels = [0] * len(corpus)

        # Threshold-based multi-label prediction
        threshold = 0.5
        for idx, score in enumerate(scores):
            if score >= threshold:
                predicted_labels[idx] = 1

        y_true.append(true_labels)
        y_pred.append(predicted_labels)

    # Calculate multi-label metrics
    hamming = hamming_loss(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='samples')
    recall = recall_score(y_true, y_pred, average='samples')
    f1 = f1_score(y_true, y_pred, average='samples')

    print(f"\nEvaluation for {model_name}:")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


if __name__ == '__main__':
    train, val, label_map, raw = build_dataset()
    corpus = []
    for entry in raw:
        name = entry.get('disease_name') or entry.get('name') or 'Unknown'
        desc = entry.get('description') or ' '.join([entry.get(k, '') for k in ('leaf_symptoms','disease_conditions','fruit_effects')])
        label_vector = [0] * len(label_map)
        label_id = label_map.get(name)
        if label_id is not None:
            label_vector[label_id] = 1
        corpus.append({'name': name, 'description': desc, 'label_vector': label_vector})

    models = []
    try:
        m_local = SentenceTransformer('model')
        models.append(('local_model', m_local))
    except Exception as e:
        print('local model not loadable:', e)
    try:
        m_base = SentenceTransformer('all-MiniLM-L6-v2')
        models.append(('all-MiniLM-L6-v2', m_base))
    except Exception as e:
        print('hub base not loadable:', e)
    try:
        m_ft = SentenceTransformer('model_finetuned')
        models.append(('model_finetuned', m_ft))
    except Exception as e:
        print('finetuned model not loadable:', e)

    for model_name, model in models:
        evaluate_model(model_name, model, val, corpus)
