from sentence_transformers import SentenceTransformer, util
import json, random
import torch

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


def evaluate_model(model_name, model, val_set, corpus, topk=(1,3)):
    corpus_embeddings = model.encode([c['description'] for c in corpus], convert_to_tensor=True)
    results = {k:0 for k in topk}
    for text, label in val_set:
        q_emb = model.encode(text, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(q_emb, corpus_embeddings)[0]
        for k in topk:
            topk_idx = torch.topk(scores, k=k).indices
            found = False
            for idx in topk_idx:
                # check if the corpus entry's label matches
                entry_label = None
                name = corpus[int(idx.item())]['name']
                # map name back to label id: we'll compute label_map outside
                # so this function expects corpus entries to carry label_id
            # we'll handle after loop
    # We'll implement differently below (corpus includes label_id)


if __name__ == '__main__':
    train, val, label_map, raw = build_dataset()
    # build corpus with label ids
    corpus = []
    for entry in raw:
        name = entry.get('disease_name') or entry.get('name') or 'Unknown'
        desc = entry.get('description') or ' '.join([entry.get(k, '') for k in ('leaf_symptoms','disease_conditions','fruit_effects')])
        label_id = label_map.get(name)
        corpus.append({'name': name, 'description': desc, 'label_id': label_id})

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

    topk_vals = (1,3)
    for model_name, model in models:
        print('\nEvaluating', model_name)
        corpus_embeddings = model.encode([c['description'] for c in corpus], convert_to_tensor=True)
        counts = {k:0 for k in topk_vals}
        for text, true_label in val:
            q_emb = model.encode(text, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(q_emb, corpus_embeddings)[0]
            for k in topk_vals:
                topk_idx = torch.topk(scores, k=k).indices
                found = False
                for idx in topk_idx:
                    idx = int(idx.item())
                    if corpus[idx]['label_id'] == true_label:
                        found = True
                        break
                if found:
                    counts[k] += 1
        total = len(val)
        for k in topk_vals:
            acc = counts[k] / total
            print(f' top-{k} accuracy: {acc:.3f} ({counts[k]}/{total})')
