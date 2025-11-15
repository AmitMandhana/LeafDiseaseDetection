# Detailed Model Architecture — Transformer + Retrieval (LeafDiseaseDetection)

This document explains, in technical depth, the model architecture and training/inference steps used in this repository. It is written so you can paste sections directly into slides and speaker notes.

## Short summary (one-liner)
We use a SentenceTransformers Transformer encoder (base: all-MiniLM-L6-v2) to convert symptom text into fixed-size embeddings; during fine-tuning we add a small classification (softmax) head trained with SoftmaxLoss so that disease classes form separable clusters in embedding space; at inference we compute cosine similarity between a query embedding and pre-computed disease description embeddings and return the top match.

---

## Contract
- Inputs: free-form symptom text (strings). May be short ("yellow leaves, brown spots") or longer (multiple sentences). Inputs are passed through tokenization and then through the Transformer encoder.
- Outputs: ranked disease matches (top-k) with confidence (cosine similarity %). During training the model also outputs classification logits and losses used for optimization.
- Success criteria: a high top-1/top-3 retrieval accuracy (measured on the synthetic validation set built from `disease_data.json`).

---

## Pipeline overview (high-level)
1. Text preprocessing & tokenization
2. Token embeddings + positional encodings
3. Transformer encoder stack (multi-head self-attention + position-wise feed-forward networks repeated L times)
4. Pooling layer to obtain a fixed-size sentence embedding
5. (Training only) Small MLP classifier head + SoftmaxLoss
6. (Inference only) Cosine similarity between query embedding and corpus embeddings -> top-k retrieval

The accompanying SVG (`assets/model_architecture_detailed.svg`) visualizes these steps.

---

## Step-by-step technical explanation

### 1) Tokenization
- The base SentenceTransformers model (MiniLM) uses a subword tokenizer (often BPE or WordPiece depending on the specific pretrained weights). The tokenizer:
  - Splits input text into subword tokens.
  - Maps tokens to integer ids using the model vocabulary (embedding lookup indices).
  - Produces an attention mask marking which token positions are real input vs padding.

Why: subword tokenization handles rare words and morphological variants efficiently. It keeps vocabulary size manageable and ensures unseen words are broken into known subword pieces.

Edge cases: out-of-vocabulary tokens are not an error — they are decomposed into subwords or use an unknown token. Very long inputs are truncated by the tokenizer to the model's max sequence length (e.g., 256 / 512 tokens).


### 2) Token embeddings + positional encodings
- After tokenization each token id is converted to a token embedding vector via an embedding matrix E ∈ R^{V×d}, where V is vocabulary size and d is model hidden size.
- Transformer encoders are position-agnostic by design; to preserve word order we add positional encodings p_i to each token embedding: x_i = E[token_i] + p_i.

Mathematically, for token t_i: x_i = Embedding(t_i) + PosEnc(i)

Why: positional encodings enable the model to reason about order (e.g., "leaf" before "spots"). Learned positional embeddings or fixed sinusoidal encodings both serve the same goal.


### 3) Transformer encoder block (repeated L layers)
Each encoder layer consists of:
- Multi-Head Self-Attention (MHA)
- Add & LayerNorm (residual connection)
- Position-wise Feed-Forward Network (FFN)
- Add & LayerNorm (residual connection)

Multi-head attention details (one block):
- Input: sequence of token vectors X ∈ R^{n×d} (n tokens, d hidden size).
- For each attention head h we compute linear projections:
  Q_h = X W_h^Q,  K_h = X W_h^K,  V_h = X W_h^V
  where W^Q, W^K, W^V ∈ R^{d×d_k} are learned matrices and d_k = d / num_heads typically.
- Scaled dot-product attention (per head):
  Attention(Q_h, K_h, V_h) = softmax( (Q_h K_h^T) / sqrt(d_k) ) V_h
- Concatenate all heads and apply a final linear projection to return to dimension d.

Why scaled dot-product and multiple heads?
- Scaling by sqrt(d_k) prevents extremely large dot products that would push softmax into regions with tiny gradients.
- Multiple heads allow the model to attend to different aspects or positions in parallel (syntax, short/long-range dependencies, negation, etc.).

Residual + LayerNorm
- After attention, we add the original input back (residual) and apply LayerNorm. This stabilizes training and helps gradients flow through the deep stack.

Position-wise FFN
- A two-layer fully-connected network (applied independently to each token position) with a non-linearity (typically GeLU):
  FFN(x) = max(0, xW_1 + b_1) W_2 + b_2  (GeLU often used instead of ReLU)
- Another residual + LayerNorm follows.

Repeating L times
- The encoder stack (L layers) allows iterative refinement — each layer can model more abstract or longer-range relationships.

Typical hyperparameters (MiniLM-like):
- d (hidden size): 384 or 512 depending on variant
- L (layers): 6 (MiniLM-L6 => 6 transformer layers)
- num_heads: 12 or 6 depending on d and variant


### 4) Pooling (sentence embedding)
- The Transformer produces contextual token embeddings for each input position. We need a fixed-size vector for the whole sentence.
- SentenceTransformers commonly uses one of: CLS token embedding, mean pooling of token embeddings, or max pooling. In this codebase the pooling layer performs mean pooling over the last hidden states while respecting the attention mask (i.e., padding tokens are excluded).

Mean pooling formula:

Let H ∈ R^{n×d} be the final token embeddings and m ∈ {0,1}^n the attention mask. The sentence embedding s ∈ R^d is:

s = (Σ_{i=1..n} m_i H_i) / (Σ_i m_i)

Why mean pooling?
- It aggregates contextual information across tokens and is robust for many sentence-level tasks.
- It's simple and empirically effective for retrieval.

Normalization
- Embeddings are often L2-normalized before cosine retrieval. For retrieval, cosine similarity between embeddings a and b is:

cos_sim(a,b) = (a · b) / (||a|| ||b||)

When embeddings are L2-normalized, cosine similarity reduces to dot product.


### 5) Training head (SoftmaxLoss) — how and why
- During fine-tuning we attach a small classification head (usually a fully-connected layer) on top of the pooled embedding. The head maps the pooled embedding vector s ∈ R^d to logits z ∈ R^{C} where C is the number of disease classes.

z = W s + b  (W ∈ R^{C×d})

- SoftmaxLoss applies a softmax over logits and computes cross-entropy with the ground-truth label y ∈ {1..C}.

Loss = - log softmax(z_y)

Why train with SoftmaxLoss if we use cosine similarity at inference?
- SoftmaxLoss pulls examples of the same label closer in embedding space and pushes different-label examples apart — effectively creating clusters for each disease class. This separation improves nearest-neighbor retrieval performance even if we discard the classifier at inference and use the embedding space alone.

Implementation detail in this repo
- The training examples are created as InputExample(texts=[t, t], label=l). The duplication is an API convenience: the Softmax training loop in sentence-transformers expects pairs of texts; duplicating the text as the second element creates a straightforward single-sentence supervision signal.

Optimization & schedule
- Typical optimizer: AdamW (often used for Transformers).
- Warmup steps: a small number of steps (10 in the script) are used to ramp the learning rate from 0 to the target value to stabilize early training.
- Backprop updates both encoder parameters (optionally) and the classification head.

Regularization and stability
- Batch size and label distribution matter: if some classes have very few synthetic queries, the learned clusters may be noisy.
- Gradient clipping, weight decay, and learning rate schedules can help; this small script uses conservative defaults suitable for quick fine-tuning on CPU.


### 6) Inference: retrieval using cosine similarity
- After training, the app computes and caches embeddings for each disease description in the corpus:
  corpus_embeddings = model.encode([desc_i for each disease], convert_to_tensor=True)
- For a user query q, compute q_emb = model.encode(q, convert_to_tensor=True).
- Compute cosine similarity scores between q_emb and each corpus embedding. In the codebase this is implemented using PyTorch with:
  scores = util.pytorch_cos_sim(q_emb, corpus_embeddings)[0]
- Retrieve top-k nearest neighbors using torch.topk and return the corresponding disease entry and similarity score (often scaled to a percentage).

Why cosine similarity?
- Cosine similarity measures angular distance; it works well for embedding comparisons where absolute magnitude is less informative than direction.
- L2-normalized vectors make cosine an efficient dot product.


## Key formulas (compact)
- Scaled dot-product attention (per head):
  Attention(Q,K,V) = softmax( Q K^T / sqrt(d_k) ) V

- Mean pooling:
  s = (Σ m_i H_i) / (Σ m_i)

- Softmax & cross-entropy:
  p_j = exp(z_j) / Σ_k exp(z_k)
  Loss = - log p_{y}

- Cosine similarity:
  cos_sim(a,b) = (a · b) / (||a|| ||b||)


## Why this design (rationale)
- Use a Transformer encoder (MiniLM) because it provides contextualized token embeddings that capture semantics beyond bag-of-words.
- Fine-tune with a classification head (SoftmaxLoss) because a discriminative objective arranges the embedding space so that retrieval by nearest-neighbor performs better.
- Mean pooling is simple, robust, and fast for sentence-level embeddings on short symptom texts.

Tradeoffs
- Using SoftmaxLoss requires labelled classes and can be sensitive to class imbalance. It also adds some training cost compared to unsupervised contrastive approaches.
- We fine-tune a small model (MiniLM) for computational efficiency; larger models usually give better accuracy at the cost of compute and latency.


## Practical tips and edge cases
- Long descriptions: ensure tokenizer truncation is acceptable; consider summarizing long disease descriptions or using sliding window pooling.
- OOV / rare terms: subword tokenization mitigates this. Consider adding domain-specific tokens if you have repeated domain vocabulary.
- Small classes: augment data (more synthetic queries or paraphrases) for underrepresented diseases.
- Caching corpus embeddings: precompute and persist corpus embeddings to disk to speed up app start and to make inference real-time for many corpus entries.


## Slide / speaker notes (short)
- Use the SVG `assets/model_architecture_detailed.svg` to show the flow from raw text to prediction.
- Emphasize the difference between training (softmax head + cross-entropy) and inference (embedding retrieval via cosine similarity).
- Mention key strengths (fast, accurate retrieval with small model) and limitations (text-only, depends on corpus quality).

---

If you want, I can also:
- produce a high-resolution PNG export of the SVG for PowerPoint compatibility,
- generate a short single-slide PDF summarizing the math (attention and pooling formulas), or
- create a .pptx with 6 slides based on the earlier README slide suggestions embedding this detailed diagram.

Tell me which of these you'd like next (PNG resolution or PPTX), and whether to include speaker notes.  