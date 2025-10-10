from sentence_transformers import SentenceTransformer, util
import json
import torch

# try fine-tuned first
try:
    model = SentenceTransformer('model_finetuned')
    print('Loaded model from model_finetuned')
except Exception:
    model = SentenceTransformer('model')
    print('Loaded model from model')

with open('disease_data.json', 'r', encoding='utf-8') as f:
    raw = json.load(f)

corpus = []
for entry in raw:
    name = entry.get('disease_name') or entry.get('name') or 'Unknown'
    desc = entry.get('description') or ' '.join([entry.get(k, '') for k in ('leaf_symptoms','disease_conditions','fruit_effects')])
    corpus.append({'name': name, 'description': desc})

corpus_embeddings = model.encode([c['description'] for c in corpus], convert_to_tensor=True)

queries = [
    "Leaves are yellowing with small brown spots, fruit looks scabby",
    "My tomato leaves have white powdery stuff and some curling",
    "Young leaves have orange spots, wet weather lately",
]

for query in queries:
    q_emb = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(q_emb, corpus_embeddings)[0]
    topk = torch.topk(scores, k=5)

    print('\nQuery:', query)
    for score, idx in zip(topk[0], topk[1]):
        i = int(idx.item())
        print(f"  {corpus[i]['name']} — score: {float(score):.4f}")
