---
name: explain
description: Explain nanochat architecture with ASCII diagrams - attention, mlp, optimizer, dataloader, rotary, etc.
arguments:
  - name: module
    description: "Module to explain: attention, mlp, optimizer, dataloader, rotary, gpt, tokenizer, engine"
---

# Architecture Explainer

Deep-dive explanations of nanochat components with diagrams.

## How to Use

When user asks `/explain <module>`, read the relevant source file and explain with:
1. ASCII diagram showing data flow
2. Component-by-component breakdown
3. Comparison to standard implementations
4. Paper references

## Module: GPT (Full Model)

**Source:** `nanochat/gpt.py`

```
┌─────────────────────────────────────────────────────────────┐
│                        GPT Model                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input tokens [B, T]                                        │
│         │                                                    │
│         ▼                                                    │
│   ┌─────────────┐                                           │
│   │  Embedding  │  vocab_size → model_dim                   │
│   └─────────────┘                                           │
│         │                                                    │
│         ▼                                                    │
│   ┌─────────────────────────────────────────┐              │
│   │           Transformer Block ×N           │              │
│   │  ┌─────────────────────────────────┐    │              │
│   │  │  RMSNorm (no learnable params)  │    │              │
│   │  └─────────────────────────────────┘    │              │
│   │         │                               │              │
│   │         ▼                               │              │
│   │  ┌─────────────────────────────────┐    │              │
│   │  │  CausalSelfAttention (GQA)      │    │              │
│   │  │  + RoPE positional encoding     │    │              │
│   │  │  + QK normalization             │    │              │
│   │  │  + Value embeddings (alt layers)│    │              │
│   │  └─────────────────────────────────┘    │              │
│   │         │                               │              │
│   │         ▼  (residual connection)        │              │
│   │  ┌─────────────────────────────────┐    │              │
│   │  │  RMSNorm                         │    │              │
│   │  └─────────────────────────────────┘    │              │
│   │         │                               │              │
│   │         ▼                               │              │
│   │  ┌─────────────────────────────────┐    │              │
│   │  │  MLP with ReLU² activation      │    │              │
│   │  └─────────────────────────────────┘    │              │
│   │         │                               │              │
│   │         ▼  (residual connection)        │              │
│   └─────────────────────────────────────────┘              │
│         │                                                    │
│         ▼                                                    │
│   ┌─────────────┐                                           │
│   │   RMSNorm   │                                           │
│   └─────────────┘                                           │
│         │                                                    │
│         ▼                                                    │
│   ┌─────────────┐                                           │
│   │  LM Head    │  model_dim → vocab_size (weight tied)     │
│   └─────────────┘                                           │
│         │                                                    │
│         ▼                                                    │
│   Output logits [B, T, vocab_size]                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key design choices:**
- `model_dim = depth × aspect_ratio` (default 64)
- RMSNorm without learnable parameters
- Weight tying between embedding and LM head

---

## Module: Attention

**Source:** `nanochat/gpt.py` - `CausalSelfAttention` class

```
Input x: [B, T, D]
    │
    ├────────────────┬────────────────┐
    │                │                │
    ▼                ▼                ▼
┌───────┐      ┌───────┐        ┌───────┐
│ W_q   │      │ W_k   │        │ W_v   │
│D→n_h×d│      │D→n_kv×d│       │D→n_kv×d│
└───────┘      └───────┘        └───────┘
    │                │                │
    ▼                ▼                ▼
   Q                K                V
[B,T,n_h,d]    [B,T,n_kv,d]    [B,T,n_kv,d]
    │                │                │
    ▼                ▼                │
┌─────────┐    ┌─────────┐           │
│ QK Norm │    │ QK Norm │           │
└─────────┘    └─────────┘           │
    │                │                │
    ▼                ▼                │
┌─────────┐    ┌─────────┐           │
│  RoPE   │    │  RoPE   │           │
└─────────┘    └─────────┘           │
    │                │                │
    └───────┬────────┘                │
            │                         │
            ▼                         ▼
     ┌─────────────────────────────────┐
     │   Scaled Dot-Product Attention   │
     │   (with causal mask + GQA)       │
     │                                  │
     │   Q @ K.T / sqrt(d) → softmax    │
     │   → @ V                          │
     └─────────────────────────────────┘
                    │
                    ▼
              ┌───────────┐
              │    W_o    │
              │n_h×d → D  │
              └───────────┘
                    │
                    ▼
            Output: [B, T, D]
```

**GQA (Group Query Attention):**
- `n_heads = depth` (query heads)
- `n_kv_heads = max(1, depth // 4)` (key/value heads)
- Reduces KV cache size by 4× for inference

**QK Normalization:**
- Normalizes Q and K before attention
- Stabilizes training at large scale
- From "Scaling Vision Transformers" paper

---

## Module: MLP

**Source:** `nanochat/gpt.py` - `MLP` class

```
Input x: [B, T, D]
        │
        ▼
   ┌─────────┐
   │  W_gate │  D → 4D (or hidden_dim)
   └─────────┘
        │
        ▼
     gate
        │
        ├─────────────────┐
        │                 │
        ▼                 ▼
   ┌─────────┐      ┌─────────┐
   │  ReLU²  │      │  W_up   │
   └─────────┘      └─────────┘
        │                 │
        │     x: [B,T,4D] │
        └────────┬────────┘
                 │
                 ▼
            element-wise ×
                 │
                 ▼
           ┌─────────┐
           │  W_down │  4D → D
           └─────────┘
                 │
                 ▼
          Output: [B, T, D]
```

**ReLU² (Squared ReLU):**
```python
def relu_squared(x):
    return torch.relu(x) ** 2
```
- Sparser activations than GELU
- Better gradient flow than plain ReLU
- From "Primer" paper

**SwiGLU variant:**
- Uses gating mechanism: `(gate * relu²(x)) @ W_down`
- 4× expansion ratio (can be configured)

---

## Module: RoPE (Rotary Position Embeddings)

**Source:** `nanochat/gpt.py` - `apply_rotary_emb`

```
Position-based rotation of Q, K vectors

For each position t and dimension pair (i, i+1):

   ┌─────────────────────────────────────────┐
   │  cos(θ_i × t)  -sin(θ_i × t)            │
   │  sin(θ_i × t)   cos(θ_i × t)            │
   └─────────────────────────────────────────┘
              ×
   ┌─────────────────────────────────────────┐
   │  [q_i, q_{i+1}]                         │
   └─────────────────────────────────────────┘

Where θ_i = 10000^(-2i/d)
```

**Benefits:**
- No learned positional parameters
- Relative position encoded via rotation
- Extrapolates better than learned positions

---

## Module: Optimizer

**Source:** `nanochat/optim.py`

```
┌─────────────────────────────────────────────────────────────┐
│                    MuonAdamW Optimizer                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Parameters                                                 │
│        │                                                     │
│        ├──────────────────┬─────────────────────┐           │
│        │                  │                     │           │
│        ▼                  ▼                     ▼           │
│   ┌─────────┐       ┌─────────┐          ┌─────────┐       │
│   │Embedding│       │ Attention│          │   MLP   │       │
│   │ params  │       │  W_q,k,v │          │  W_*    │       │
│   └─────────┘       └─────────┘          └─────────┘       │
│        │                  │                     │           │
│        ▼                  ▼                     ▼           │
│   ┌─────────┐       ┌─────────────────────────────┐        │
│   │  AdamW  │       │           Muon              │        │
│   │         │       │  (for 2D matrix params)     │        │
│   │ LR/√dim │       │                             │        │
│   └─────────┘       └─────────────────────────────┘        │
│        │                          │                         │
│        └──────────┬───────────────┘                        │
│                   │                                         │
│                   ▼                                         │
│             Updated θ                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Muon Optimizer:**
- Uses Newton-Schulz orthogonalization
- More efficient for matrix parameters
- From "Muon" paper

**AdamW scaling:**
- LR scales by `1/√model_dim`
- Weight decay scales by `1/depth²`

---

## Module: Dataloader

**Source:** `nanochat/dataloader.py`

```
┌─────────────────────────────────────────────────────────────┐
│                    Distributed Dataloader                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   FineWeb-Edu Parquet Shards                                │
│   [shard_0.parquet] [shard_1.parquet] ... [shard_N.parquet] │
│           │               │                    │            │
│           └───────────────┼────────────────────┘            │
│                           │                                  │
│                           ▼                                  │
│                    ┌─────────────┐                          │
│                    │  Shard      │                          │
│                    │  Assignment │  (per rank)              │
│                    └─────────────┘                          │
│                           │                                  │
│                           ▼                                  │
│                    ┌─────────────┐                          │
│                    │  Tokenize   │  (BPE)                   │
│                    └─────────────┘                          │
│                           │                                  │
│                           ▼                                  │
│                    ┌─────────────────────────┐              │
│                    │  Best-Fit Packing       │              │
│                    │  (BOS-aligned)          │              │
│                    │                         │              │
│                    │  Pack multiple docs     │              │
│                    │  into seq_len chunks    │              │
│                    └─────────────────────────┘              │
│                           │                                  │
│                           ▼                                  │
│                    ┌─────────────┐                          │
│                    │   Batch     │  [B, seq_len]            │
│                    └─────────────┘                          │
│                           │                                  │
│                           ▼                                  │
│                       To GPU                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Best-Fit Packing:**
- Packs multiple documents to minimize padding
- Each document starts with BOS token
- Attention mask handles document boundaries

---

## Module: Tokenizer

**Source:** `nanochat/tokenizer.py`

```
┌─────────────────────────────────────────────────────────────┐
│                       BPE Tokenizer                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Text: "Hello, world!"                                      │
│              │                                               │
│              ▼                                               │
│   ┌─────────────────────┐                                   │
│   │  Pre-tokenization   │  (split on spaces, punctuation)   │
│   └─────────────────────┘                                   │
│              │                                               │
│              ▼                                               │
│   ["Hello", ",", " ", "world", "!"]                         │
│              │                                               │
│              ▼                                               │
│   ┌─────────────────────┐                                   │
│   │   BPE Encoding      │  (merge frequent pairs)           │
│   │   via tiktoken      │                                   │
│   └─────────────────────┘                                   │
│              │                                               │
│              ▼                                               │
│   Token IDs: [15496, 11, 220, 995, 0]                       │
│                                                              │
│   Special tokens:                                            │
│   - BOS: <|begin_of_text|>                                  │
│   - EOS: <|end_of_text|>                                    │
│   - PAD: <|finetune_pad|>                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**GPT-4 compatible tokenizer:**
- Uses tiktoken for fast inference
- rustbpe for training
- ~100k vocabulary

---

## Module: Engine

**Source:** `nanochat/engine.py`

```
┌─────────────────────────────────────────────────────────────┐
│                    Inference Engine                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Prompt: "Why is the sky blue?"                            │
│              │                                               │
│              ▼                                               │
│   ┌─────────────────────┐                                   │
│   │     Tokenize        │                                   │
│   └─────────────────────┘                                   │
│              │                                               │
│              ▼                                               │
│   ┌─────────────────────────────────────────┐              │
│   │           KV Cache Manager              │              │
│   │                                         │              │
│   │  Stores key/value tensors for each     │              │
│   │  layer to avoid recomputation          │              │
│   │                                         │              │
│   │  [Layer 0 K,V] [Layer 1 K,V] ... [N]  │              │
│   └─────────────────────────────────────────┘              │
│              │                                               │
│              ▼                                               │
│   ┌─────────────────────────────────────────┐              │
│   │        Autoregressive Loop              │              │
│   │                                         │              │
│   │  for _ in range(max_tokens):           │              │
│   │    logits = model(tokens, kv_cache)    │              │
│   │    next_token = sample(logits)         │              │
│   │    tokens.append(next_token)           │              │
│   │    if next_token == EOS: break         │              │
│   └─────────────────────────────────────────┘              │
│              │                                               │
│              ▼                                               │
│   ┌─────────────────────┐                                   │
│   │  Sampling Strategy  │                                   │
│   │  - Temperature      │                                   │
│   │  - Top-k            │                                   │
│   └─────────────────────┘                                   │
│              │                                               │
│              ▼                                               │
│   Output: "The sky appears blue because..."                 │
│                                                              │
│   ┌─────────────────────────────────────────┐              │
│   │         Tool Support                    │              │
│   │  Built-in Python calculator:            │              │
│   │  <<2+2>> → 4                            │              │
│   └─────────────────────────────────────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Paper References

| Component | Paper |
|-----------|-------|
| RoPE | "RoFormer: Enhanced Transformer with Rotary Position Embedding" |
| GQA | "GQA: Training Generalized Multi-Query Transformer Models" |
| QK Norm | "Scaling Vision Transformers to 22 Billion Parameters" |
| ReLU² | "Primer: Searching for Efficient Transformers" |
| Muon | "Muon: Momentum Orthogonalized Update for Neural Networks" |
| RMSNorm | "Root Mean Square Layer Normalization" |
| Flash Attention | "FlashAttention: Fast and Memory-Efficient Attention" |
