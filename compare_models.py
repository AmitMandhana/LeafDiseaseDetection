from sentence_transformers import SentenceTransformer, util
import json
import torch

models_to_check = []
# try local original model
try:
    m_local = SentenceTransformer('model')
    models_to_check.append(('local_model', m_local))
except Exception as e:
    print('local model not loadable:', e)

# hub base
try:
    m_base = SentenceTransformer('all-MiniLM-L6-v2')
    models_to_check.append(('all-MiniLM-L6-v2', m_base))
except Exception as e:
    print('hub base not loadable:', e)

# fine-tuned
try:
    m_ft = SentenceTransformer('model_finetuned')
    models_to_check.append(('model_finetuned', m_ft))
except Exception as e:
    print('finetuned model not loadable:', e)

with open('disease_data.json', 'r', encoding='utf-8') as f:
    raw = json.load(f)

corpus = []
for entry in raw:
    name = entry.get('disease_name') or entry.get('name') or 'Unknown'
    desc = entry.get('description') or ' '.join([entry.get(k, '') for k in ('leaf_symptoms','disease_conditions','fruit_effects')])
    corpus.append({'name': name, 'description': desc})

def ensemble_predictions(models, query, corpus_embeddings):
    combined_scores = None
    for model_name, model in models:
        q_emb = model.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(q_emb, corpus_embeddings)[0]
        if combined_scores is None:
            combined_scores = scores
        else:
            combined_scores += scores
    # Average the scores across models
    combined_scores /= len(models)
    return combined_scores

queries = [
    ("Leaves are yellowing with small brown spots, fruit looks scabby", 'expected: Tomato Bacterial Spot or Pepper Bacterial Spot'),
    ("My tomato leaves have white powdery stuff and some curling", 'expected: Tomato Powdery Mildew or Powdery Mildew'),
    ("Young leaves have orange spots, wet weather lately", 'expected: Cedar Apple Rust or rust-type disease'),
]

for model_name, model in models_to_check:
    print('\n--- MODEL:', model_name, '---')
    corpus_embeddings = model.encode([c['description'] for c in corpus], convert_to_tensor=True)
    for q, note in queries:
        q_emb = model.encode(q, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(q_emb, corpus_embeddings)[0]
        topk = torch.topk(scores, k=3)
        print('\nQuery:', q)
        print('  Note:', note)
        for score, idx in zip(topk[0], topk[1]):
            i = int(idx.item())
            print(f"   {corpus[i]['name']} — {float(score):.4f}")

# Ensemble evaluation
print('\n--- ENSEMBLE MODEL ---')
corpus_embeddings = [model.encode([c['description'] for c in corpus], convert_to_tensor=True) for _, model in models_to_check]
for q, note in queries:
    combined_scores = ensemble_predictions(models_to_check, q, corpus_embeddings)
    topk = torch.topk(combined_scores, k=3)
    print('\nQuery:', q)
    print('  Note:', note)
    for score, idx in zip(topk[0], topk[1]):
        i = int(idx.item())
        print(f"   {corpus[i]['name']} — {float(score):.4f}")
