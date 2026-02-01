# nanochat Code Walkthrough

> A detailed guide mapping the Transformer architecture to nanochat's implementation.
> Target audience: Advanced undergrads/grads with PyTorch experience and basic Transformer knowledge.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [The GPT Model (`nanochat/gpt.py`)](#2-the-gpt-model)
3. [Attention Mechanism Deep Dive](#3-attention-mechanism-deep-dive)
4. [MLP Block](#4-mlp-block)
5. [The Transformer Block](#5-the-transformer-block)
6. [Tokenization (`nanochat/tokenizer.py`)](#6-tokenization)
7. [Data Loading (`nanochat/dataloader.py`)](#7-data-loading)
8. [The Optimizer (`nanochat/optim.py`)](#8-the-optimizer)
9. [Inference Engine (`nanochat/engine.py`)](#9-inference-engine)
10. [Training Loop (`scripts/base_train.py`)](#10-training-loop)
11. [Complete Data Flow](#11-complete-data-flow)

---

## 1. High-Level Architecture

nanochat implements a **decoder-only Transformer** (like GPT) with modern optimizations. Here's the complete system:

```
                           NANOCHAT SYSTEM OVERVIEW
 +=============================================================================+
 |                                                                             |
 |   +------------------+     +------------------+     +------------------+    |
 |   |                  |     |                  |     |                  |    |
 |   |   Raw Text       |---->|   Tokenizer      |---->|   Token IDs      |    |
 |   |   "Hello world"  |     |   (BPE)          |     |   [1, 234, 56]   |    |
 |   |                  |     |                  |     |                  |    |
 |   +------------------+     +------------------+     +------------------+    |
 |                                                              |              |
 |                                                              v              |
 |   +------------------+     +------------------+     +------------------+    |
 |   |                  |     |                  |     |                  |    |
 |   |   DataLoader     |---->|   Batched        |---->|   GPT Model      |    |
 |   |   (Best-fit)     |     |   Tensors        |     |   (Transformer)  |    |
 |   |                  |     |   [B, T]         |     |                  |    |
 |   +------------------+     +------------------+     +------------------+    |
 |                                                              |              |
 |                                                              v              |
 |   +------------------+     +------------------+     +------------------+    |
 |   |                  |     |                  |     |                  |    |
 |   |   MuonAdamW      |<----|   Gradients      |<----|   Logits         |    |
 |   |   Optimizer      |     |                  |     |   [B, T, V]      |    |
 |   |                  |     |                  |     |                  |    |
 |   +------------------+     +------------------+     +------------------+    |
 |                                                                             |
 +=============================================================================+

 B = Batch size, T = Sequence length, V = Vocab size (32768)
```

---

## 2. The GPT Model

### 2.1 Model Configuration

The model is configured via `GPTConfig` dataclass:

```
                        GPTConfig PARAMETERS
 +-----------------------------------------------------------------------+
 |  Parameter      | Default  | Description                             |
 |-----------------|----------|------------------------------------------
 |  sequence_len   | 2048     | Maximum context window                  |
 |  vocab_size     | 32768    | Number of tokens in vocabulary          |
 |  n_layer        | 12       | Number of transformer blocks (depth)    |
 |  n_head         | 6        | Number of query attention heads         |
 |  n_kv_head      | 6        | Number of key/value heads (for GQA)     |
 |  n_embd         | 768      | Model dimension (d_model)               |
 |  window_pattern | "SSSL"   | Sliding window pattern per layer        |
 +-----------------------------------------------------------------------+

 Key relationship: n_embd = depth * aspect_ratio (default aspect_ratio = 64)

 Example: depth=12 --> n_embd = 12 * 64 = 768
          depth=24 --> n_embd = 24 * 64 = 1536
```

### 2.2 Model Architecture Overview

```
                         GPT MODEL ARCHITECTURE

     Input Token IDs: [B, T]
              |
              v
     +------------------+
     |  Token Embedding |  wte: Embedding(vocab_size, n_embd)
     |  (wte)           |  Looks up dense vectors for each token
     +------------------+
              |
              v
     +------------------+
     |    RMSNorm       |  norm(x): Normalize embedding
     +------------------+  No learnable params!
              |
              |<----------------------------------------+
              v                                         |
     +==================+                               |
     ||  Transformer   ||                               |
     ||    Block 0     ||  Each block has:              | x0 residual
     ||                ||  - CausalSelfAttention        | connection
     ||  Attn -> MLP   ||  - MLP                        |
     +==================+                               |
              |                                         |
              | x = resid_lambda * x + x0_lambda * x0   |
              v                                         |
     +==================+                               |
     ||  Transformer   ||                               |
     ||    Block 1     ||                               |
     +==================+                               |
              |                                         |
              :  (repeat n_layer times)                 |
              :                                         |
              v                                         |
     +==================+                               |
     ||  Transformer   ||                               |
     ||  Block n-1     ||                               |
     +==================+---------<---------------------+
              |
              v
     +------------------+
     |    RMSNorm       |  Final normalization
     +------------------+
              |
              v
     +------------------+
     |   LM Head        |  Linear(n_embd, vocab_size)
     |   (lm_head)      |  Projects to vocabulary logits
     +------------------+
              |
              v
     +------------------+
     |   Logit Softcap  |  15 * tanh(logits / 15)
     +------------------+  Smoothly caps logits to [-15, 15]
              |
              v
     Output Logits: [B, T, vocab_size]
```

### 2.3 Code Mapping: `GPT.__init__`

```python
# nanochat/gpt.py, line 146-186

class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config

        # Sliding window sizes for each layer
        self.window_sizes = self._compute_window_sizes(config)

        # The core transformer components
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),  # Token embedding
            "h": nn.ModuleList([                                    # Transformer blocks
                Block(config, layer_idx)
                for layer_idx in range(config.n_layer)
            ]),
        })

        # Output projection (vocabulary prediction)
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)

        # Per-layer learnable scalars for residual connections
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # Scales residual
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # Blends in initial embed

        # Value embeddings for alternating layers (ResFormer-style)
        self.value_embeds = nn.ModuleDict({...})

        # Precomputed rotary embeddings (RoPE)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
```

### 2.4 The Forward Pass

```python
# nanochat/gpt.py, line 388-423

def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
    B, T = idx.size()

    # Get rotary embeddings for current sequence
    cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

    # Token embedding + normalization
    x = self.transformer.wte(idx)   # [B, T] -> [B, T, n_embd]
    x = norm(x)                      # RMSNorm (no learnable params)
    x0 = x                           # Save for x0 residual connection

    # Process through all transformer blocks
    for i, block in enumerate(self.transformer.h):
        # Apply learnable residual scaling
        x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0

        # Get value embeddings if this layer uses them
        ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None

        # Forward through block (attention + MLP)
        x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)

    x = norm(x)  # Final normalization

    # Compute logits with softcap
    softcap = 15
    logits = self.lm_head(x)
    logits = logits[..., :self.config.vocab_size]  # Remove padding
    logits = softcap * torch.tanh(logits / softcap)

    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        return loss
    return logits
```

---

## 3. Attention Mechanism Deep Dive

### 3.1 Attention Architecture

```
                    CAUSAL SELF-ATTENTION

 Input x: [B, T, C]    (C = n_embd = 768)
        |
        +------------+------------+------------+
        |            |            |            |
        v            v            v            v
   +--------+   +--------+   +--------+   +--------+
   | c_q    |   | c_k    |   | c_v    |   | ve     |
   | Linear |   | Linear |   | Linear |   | Embed  |
   +--------+   +--------+   +--------+   +--------+
        |            |            |            |
        v            v            v            v
   Q:[B,T,H,D]  K:[B,T,Hkv,D] V:[B,T,Hkv,D]  VE:[B,T,Hkv,D]
        |            |            |            |
        |            |            +-----+------+
        |            |                  | Value + VE mixing
        v            v                  v
   +---------+  +---------+        +---------+
   | RoPE    |  | RoPE    |        | Gate    |
   | (cos,   |  | (cos,   |        | Mixing  |
   |  sin)   |  |  sin)   |        +---------+
   +---------+  +---------+             |
        |            |                  v
        v            v            V_final:[B,T,Hkv,D]
   +---------+  +---------+             |
   | QK Norm |  | QK Norm |             |
   +---------+  +---------+             |
        |            |                  |
        +------+-----+                  |
               |                        |
               v                        v
        +------------------------------------------+
        |          Flash Attention                 |
        |   softmax(Q @ K^T / sqrt(d)) @ V        |
        |   with causal mask + sliding window     |
        +------------------------------------------+
                           |
                           v
                    Y: [B, T, H, D]
                           |
                           v
                    +------------+
                    | c_proj     |  Linear projection back to n_embd
                    | [H*D -> C] |
                    +------------+
                           |
                           v
                    Output: [B, T, C]

 H = n_head (query heads)
 Hkv = n_kv_head (key/value heads, for GQA)
 D = head_dim = n_embd / n_head
```

### 3.2 Rotary Position Embeddings (RoPE)

RoPE encodes position by rotating pairs of dimensions in Q and K:

```
                         ROTARY EMBEDDINGS

 For a vector x = [x1, x2, x3, x4, ...] at position p:

     +-----+-----+       +-----+-----+
     | x1  | x2  |  -->  | x1' | x2' |   where x1' = x1*cos(p*f) - x2*sin(p*f)
     +-----+-----+       +-----+-----+         x2' = x1*sin(p*f) + x2*cos(p*f)

     +-----+-----+       +-----+-----+
     | x3  | x4  |  -->  | x3' | x4' |   same rotation with different frequency
     +-----+-----+       +-----+-----+

 Each pair of dimensions rotates at a different frequency:

   freq_i = 1 / (base^(2i/d))   where base=10000, d=head_dim

 Position p=0:  No rotation (cos=1, sin=0)
 Position p=1:  Small rotation
 Position p=100: Larger rotation (wraps around for high frequencies)
```

**Code Implementation:**

```python
# nanochat/gpt.py, line 51-57

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # [B, T, H, D]
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # Split into pairs
    y1 = x1 * cos + x2 * sin         # Rotate first half
    y2 = x1 * (-sin) + x2 * cos      # Rotate second half
    return torch.cat([y1, y2], 3)

# Precomputation (line 243-258)
def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
    # Frequencies: 1/base^(2i/d) for each dimension pair
    channel_range = torch.arange(0, head_dim, 2)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))

    # Positions: 0, 1, 2, ..., seq_len-1
    t = torch.arange(seq_len)

    # Outer product: [seq_len] x [head_dim/2] -> [seq_len, head_dim/2]
    freqs = torch.outer(t, inv_freq)

    # Cos and sin for rotation
    cos, sin = freqs.cos(), freqs.sin()
    return cos, sin
```

### 3.3 QK-Normalization

After applying RoPE, Q and K are normalized to stabilize training:

```python
# nanochat/gpt.py, line 94

q, k = norm(q), norm(k)  # RMSNorm on each head separately
```

This prevents attention scores from exploding during training.

### 3.4 Value Embeddings (ResFormer-style)

On alternating layers, the value vector V is augmented with a learned embedding:

```
 Standard Attention:           With Value Embeddings:

 V = Linear(x)                 V = Linear(x)
      |                             |
      v                             v
   Attention                   VE = Embedding(token_ids)
                                    |
                                    v
                               gate = sigmoid(Linear(x[:32])) * 2
                                    |
                                    v
                               V_final = V + gate * VE
                                    |
                                    v
                                 Attention
```

**Code:**

```python
# nanochat/gpt.py, line 86-89

if ve is not None:
    ve = ve.view(B, T, self.n_kv_head, self.head_dim)
    # Gate uses first 32 channels, outputs per-head scalar in (0, 2)
    gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    v = v + gate.unsqueeze(-1) * ve  # Blend in value embedding
```

### 3.5 Sliding Window Attention

nanochat uses a pattern-based sliding window to reduce memory:

```
           SLIDING WINDOW ATTENTION PATTERNS

 Pattern "SSSL" (default) tiled across layers:

 Layer 0: S (Short) - attend to last T/2 tokens only
 Layer 1: S (Short)
 Layer 2: S (Short)
 Layer 3: L (Long)  - attend to all T tokens
 Layer 4: S (Short)
 ...
 Final layer: Always L (full context)

 For T=2048, S=1024:

 Full Context (L):              Sliding Window (S):
 +---+---+---+---+---+         +---+---+---+---+---+
 | 1 | 0 | 0 | 0 | 0 |         | 1 | 0 | 0 | 0 | 0 |
 | 1 | 1 | 0 | 0 | 0 |         | 1 | 1 | 0 | 0 | 0 |
 | 1 | 1 | 1 | 0 | 0 |         | 0 | 1 | 1 | 0 | 0 |  <- Can only see
 | 1 | 1 | 1 | 1 | 0 |         | 0 | 0 | 1 | 1 | 0 |     last window_size
 | 1 | 1 | 1 | 1 | 1 |         | 0 | 0 | 0 | 1 | 1 |     tokens
 +---+---+---+---+---+         +---+---+---+---+---+
   ^                             ^
   Standard causal               Sliding window causal
```

---

## 4. MLP Block

### 4.1 Architecture

```
                      MLP (Feed-Forward)

 Input: [B, T, C]   (C = n_embd = 768)
        |
        v
 +----------------+
 |    c_fc        |  Linear(C, 4*C) - Expand to 4x width
 +----------------+
        |
        v
 +----------------+
 |    ReLU^2      |  F.relu(x).square() - ReLU squared activation
 +----------------+
        |
        v
 +----------------+
 |    c_proj      |  Linear(4*C, C) - Project back to original size
 +----------------+
        |
        v
 Output: [B, T, C]

 Note: ReLU^2 is sparser than GELU, helps with training stability
```

### 4.2 Code

```python
# nanochat/gpt.py, line 121-131

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)       # [B, T, C] -> [B, T, 4C]
        x = F.relu(x).square() # ReLU^2: sparsity + non-linearity
        x = self.c_proj(x)     # [B, T, 4C] -> [B, T, C]
        return x
```

---

## 5. The Transformer Block

### 5.1 Architecture

```
                    TRANSFORMER BLOCK

 Input x: [B, T, C]
        |
        +----------------------------------------+
        |                                        |
        v                                        |
 +----------------+                              |
 |    RMSNorm     |  norm(x)                     |
 +----------------+                              |
        |                                        |
        v                                        |
 +----------------+                              |
 |   Attention    |  CausalSelfAttention         |
 +----------------+                              |
        |                                        |
        +<---------------------------------------+
        | Add (residual connection)
        v
        +----------------------------------------+
        |                                        |
        v                                        |
 +----------------+                              |
 |    RMSNorm     |  norm(x)                     |
 +----------------+                              |
        |                                        |
        v                                        |
 +----------------+                              |
 |      MLP       |  Feed-forward                |
 +----------------+                              |
        |                                        |
        +<---------------------------------------+
        | Add (residual connection)
        v
 Output: [B, T, C]

 This is "Pre-LN" architecture: Norm BEFORE each sub-layer
```

### 5.2 Code

```python
# nanochat/gpt.py, line 134-143

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # Attention with residual (Pre-LN)
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)

        # MLP with residual (Pre-LN)
        x = x + self.mlp(norm(x))

        return x
```

---

## 6. Tokenization

### 6.1 BPE Tokenizer Overview

```
                    TOKENIZATION PIPELINE

 Raw Text                  Split                    BPE Merge
 "Hello world!"  --->  ["Hello", " world", "!"]  --->  [15496, 995, 0]
                              |
                              v
                    GPT-4 regex pattern splits on:
                    - Contractions ('s, 'll, 've, etc.)
                    - Word boundaries
                    - Numbers (1-2 digits)
                    - Punctuation
                    - Whitespace
```

### 6.2 Special Tokens

```
 +------------------+--------------------------------------------------+
 | Token            | Purpose                                          |
 +------------------+--------------------------------------------------+
 | <|bos|>          | Beginning of sequence (document delimiter)       |
 | <|user_start|>   | Start of user message in chat                   |
 | <|user_end|>     | End of user message                             |
 | <|assistant_start|> | Start of assistant response                  |
 | <|assistant_end|>   | End of assistant response                    |
 | <|python_start|> | Assistant invokes calculator tool               |
 | <|python_end|>   | End of calculator expression                    |
 | <|output_start|> | Calculator output begins                        |
 | <|output_end|>   | Calculator output ends                          |
 +------------------+--------------------------------------------------+
```

### 6.3 Conversation Rendering

```
 User: "What is 2+2?"
 Assistant: "Let me calculate: <python>2+2</python> = <output>4</output>. The answer is 4."

 Tokenized:

 <|bos|><|user_start|>What is 2+2?<|user_end|>
 <|assistant_start|>Let me calculate: <|python_start|>2+2<|python_end|>
 <|output_start|>4<|output_end|>. The answer is 4.<|assistant_end|>

 Training mask (1 = supervised, 0 = not):
 0       0              0000000000000  0
 0                 111111111111111111111 1111111111111111111111111111
                   ^                                                ^
                   Assistant generates these tokens (supervised)     |
                                                                  end token
```

---

## 7. Data Loading

### 7.1 BOS-Aligned Best-Fit Packing

```
                    DOCUMENT PACKING

 Problem: Documents have varying lengths. How to batch efficiently?

 Solution: Best-Fit Packing with BOS alignment

 Documents in buffer:
 +--------+  +------------------+  +----+  +------+
 | Doc A  |  |     Doc B        |  |Doc C|  |Doc D |
 | (500)  |  |     (1200)       |  |(300)|  |(400) |
 +--------+  +------------------+  +----+  +------+

 Packing into rows of T+1=2049 tokens:

 Row 0: [BOS][======Doc B (1200)======][BOS][Doc A (500)][BOS][Doc C (300)][crop...]
        |<------------------------ exactly 2049 tokens ------------------------>|

 Row 1: [BOS][Doc D (400)][BOS][...next docs...][...crop to fill exactly...]

 Algorithm:
 1. For each row, find LARGEST document that fits
 2. Repeat until nothing fits
 3. When stuck, crop shortest document to fill exactly

 Properties:
 - Every row starts with BOS
 - 100% utilization (no padding)
 - ~35% tokens lost to cropping (acceptable tradeoff for cleaner training)
```

### 7.2 Data Flow

```
                    DATALOADER PIPELINE

 +------------------+
 | Parquet Files    |  FineWeb-Edu dataset from HuggingFace
 | (train/*.parquet)|
 +------------------+
         |
         v
 +------------------+
 | Row Group Reader |  Read documents in batches of 128
 +------------------+
         |
         v
 +------------------+
 | Tokenizer        |  Parallel BPE encoding (8 threads)
 | (multi-threaded) |  Prepend BOS to each document
 +------------------+
         |
         v
 +------------------+
 | Document Buffer  |  Buffer of ~1000 tokenized documents
 +------------------+
         |
         v
 +------------------+
 | Best-Fit Packer  |  Pack documents into fixed-length rows
 +------------------+
         |
         v
 +------------------+
 | Batch Assembly   |  Assemble B rows into [B, T] tensor
 +------------------+
         |
         v
 +------------------+
 | GPU Transfer     |  Pinned memory -> GPU (async)
 +------------------+
         |
         v
 (inputs, targets)   inputs[i] predicts targets[i] = inputs[i+1]
```

### 7.3 Code

```python
# nanochat/dataloader.py, line 120-160 (simplified)

while True:
    rows = []
    for _ in range(B):  # Build B rows
        row = []
        while len(row) < row_capacity:  # T+1 tokens per row
            # Refill buffer if needed
            while len(doc_buffer) < buffer_size:
                refill_buffer()

            remaining = row_capacity - len(row)

            # Find largest document that fits entirely
            best_idx = -1
            best_len = 0
            for i, doc in enumerate(doc_buffer):
                if len(doc) <= remaining and len(doc) > best_len:
                    best_idx = i
                    best_len = len(doc)

            if best_idx >= 0:
                # Use the document that fits best
                doc = doc_buffer.pop(best_idx)
                row.extend(doc)
            else:
                # Nothing fits - crop shortest document
                shortest_idx = min(range(len(doc_buffer)),
                                   key=lambda i: len(doc_buffer[i]))
                doc = doc_buffer.pop(shortest_idx)
                row.extend(doc[:remaining])  # Crop to fit exactly

        rows.append(row[:row_capacity])

    # Convert to tensor and split into input/target
    row_data = torch.tensor(rows)           # [B, T+1]
    inputs = row_data[:, :-1]               # [B, T] - predict next token
    targets = row_data[:, 1:]               # [B, T] - ground truth next token
    yield inputs, targets
```

---

## 8. The Optimizer

### 8.1 MuonAdamW: A Hybrid Optimizer

nanochat uses different optimizers for different parameter types:

```
                    OPTIMIZER ASSIGNMENT

 +------------------+------------------+------------------+
 | Parameter Type   | Optimizer        | Why?             |
 +------------------+------------------+------------------+
 | Token embedding  | AdamW            | Not matrix ops   |
 | (wte)            | lr=0.2 scaled    |                  |
 +------------------+------------------+------------------+
 | Value embeddings | AdamW            | Not matrix ops   |
 | (ve)             | lr=0.2 scaled    |                  |
 +------------------+------------------+------------------+
 | LM head          | AdamW            | Output layer     |
 | (lm_head)        | lr=0.004 scaled  |                  |
 +------------------+------------------+------------------+
 | Scalars          | AdamW            | 1D parameters    |
 | (lambdas)        | lr=0.5 scaled    |                  |
 +------------------+------------------+------------------+
 | Attention/MLP    | Muon             | Matrix params    |
 | matrices         | lr=0.02          | benefit from     |
 | (c_q,c_k,c_v,    |                  | orthogonalization|
 | c_proj,c_fc)     |                  |                  |
 +------------------+------------------+------------------+

 LR scaling: AdamW params scale as 1/sqrt(d_model/768)
```

### 8.2 Muon Optimizer

Muon = **M**oment**u**m **O**rthogonalized by **N**ewton-schulz

```
                    MUON UPDATE STEP

 1. Nesterov Momentum
    +------------------+
    | momentum_buffer  |  m = beta * m + (1-beta) * grad
    +------------------+
    | Blend with grad  |  g = lerp(grad, m, beta)
    +------------------+
              |
              v

 2. Polar Express (Orthogonalization)
    +------------------+
    | Normalize        |  X = g / ||g||
    +------------------+
              |
              v
    +------------------+
    | Newton-Schulz    |  For 5 iterations:
    | Iterations       |    A = X^T @ X  (or X @ X^T)
    +------------------+    B = b*A + c*A^2
              |             X = a*X + X @ B
              v
    Result: X is approximately orthogonal (UV^T from SVD)

 3. Variance Reduction
    +------------------+
    | Per-dim variance |  Normalize variance across rows/cols
    +------------------+
              |
              v

 4. Cautious Update
    +------------------+
    | Mask by sign     |  Only update where grad and param
    | agreement        |  have same sign (cautious)
    +------------------+
              |
              v
    param = param - lr * update
```

### 8.3 Why Orthogonalization Helps

```
 Standard SGD:             Muon:

 Gradient space:           Gradient -> Orthogonal matrix:

 +--------+               +--------+
 |  .     |               |   .    |   The gradient is "spherized"
 | ...    |   --------->  |  ...   |   making updates more uniform
 |........|               | ...... |   across all directions
 +--------+               +--------+
   Uneven                   Even scale in all directions

 This helps with:
 - Faster convergence
 - Better conditioning
 - Scale invariance
```

---

## 9. Inference Engine

### 9.1 KV Cache Architecture

```
                    KV CACHE FOR EFFICIENT INFERENCE

 Without cache (naive):              With cache:

 Step 0: Process [T0]                Step 0: Process [T0]
 Step 1: Process [T0, T1]            Step 1: Process [T1] only
 Step 2: Process [T0, T1, T2]        Step 2: Process [T2] only
 ...                                 ...

 O(T^2) total computation            O(T) total computation


 KV Cache Structure:

 +----+----+----+----+----+----+----+----+
 | K0 | K1 | K2 | K3 |    |    |    |    |  <- k_cache[layer]
 +----+----+----+----+----+----+----+----+
 | V0 | V1 | V2 | V3 |    |    |    |    |  <- v_cache[layer]
 +----+----+----+----+----+----+----+----+
                     ^
                     cache_seqlens = 4 (next position to write)

 Shape: [n_layers, batch_size, max_seq_len, n_kv_heads, head_dim]
```

### 9.2 Generation with Tool Use

```
                    GENERATION STATE MACHINE

 Normal Generation:

     +--------+     sample     +----------+
     | Start  | ------------> | Generate | <----+
     +--------+               +----------+      |
                                   |            |
                              token != end      |
                                   |            |
                                   +------------+

 With Calculator Tool:

     +----------+   <|python_start|>   +---------------+
     | Generate | ------------------> | In Python     |
     +----------+                     | Block         |
          ^                           +---------------+
          |                                  |
          |                           <|python_end|>
          |                                  |
          |                                  v
          |                           +---------------+
          |                           | Evaluate      |
          |                           | Expression    |
          |                           +---------------+
          |                                  |
          |                           Force inject:
          |                           <|output_start|>result<|output_end|>
          |                                  |
          +----------------------------------+
```

### 9.3 Code Structure

```python
# nanochat/engine.py, line 164-276 (simplified)

class Engine:
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None):
        # 1. Prefill: Process entire prompt at once
        kv_cache_prefill = KVCache(batch_size=1, seq_len=len(tokens), ...)
        logits = self.model.forward(tokens, kv_cache=kv_cache_prefill)

        # 2. Clone cache for each sample
        kv_cache_decode = KVCache(batch_size=num_samples, ...)
        kv_cache_decode.prefill(kv_cache_prefill)

        # 3. Autoregressive generation loop
        while not done:
            # Sample next token
            next_ids = sample_next_token(logits, rng, temperature, top_k)

            # Handle tool use
            for i, state in enumerate(row_states):
                token = next_ids[i]

                if token == python_start:
                    state.in_python_block = True
                elif token == python_end and state.in_python_block:
                    # Evaluate and force-inject result
                    result = use_calculator(expr)
                    state.forced_tokens.extend([output_start, result, output_end])

                # Check for completion
                if token == assistant_end or token == bos:
                    state.completed = True

            yield token_column, token_masks

            # Forward single token with cache
            logits = self.model.forward(next_ids, kv_cache=kv_cache_decode)
```

---

## 10. Training Loop

### 10.1 Training Flow

```
                    TRAINING LOOP OVERVIEW

 +=============================================================+
 |                                                             |
 |  Initialization                                             |
 |  +---------------+  +---------------+  +---------------+    |
 |  | Create Model  |  | Create Optim  |  | Create Data   |    |
 |  | (meta device) |  | (MuonAdamW)   |  | Loader        |    |
 |  +---------------+  +---------------+  +---------------+    |
 |                                                             |
 +=============================================================+
                              |
                              v
 +=============================================================+
 |  Training Step (repeated num_iterations times)              |
 |                                                             |
 |  For micro_step in gradient_accumulation_steps:             |
 |     +---------------------------------------------------+   |
 |     |  1. Forward pass:  loss = model(x, y)             |   |
 |     |  2. Scale loss:    loss = loss / accum_steps      |   |
 |     |  3. Backward pass: loss.backward()                |   |
 |     |  4. Prefetch next batch (async)                   |   |
 |     +---------------------------------------------------+   |
 |                                                             |
 |  Update Learning Rate (warmup/warmdown schedule)            |
 |  Update Muon Momentum (0.85 -> 0.95 over 300 steps)        |
 |  Update Weight Decay (linear decay to 0)                    |
 |                                                             |
 |  +---------------------------------------------------+      |
 |  |  optimizer.step()  - Update all parameters        |      |
 |  |  model.zero_grad() - Clear gradients              |      |
 |  +---------------------------------------------------+      |
 |                                                             |
 +=============================================================+
                              |
                              v
 +=============================================================+
 |  Periodic Actions                                           |
 |  +---------------+  +---------------+  +---------------+    |
 |  | Eval val loss |  | CORE metric   |  | Sample text   |    |
 |  | (every 250)   |  | (every 2000)  |  | (every 2000)  |    |
 |  +---------------+  +---------------+  +---------------+    |
 |                                                             |
 +=============================================================+
```

### 10.2 Learning Rate Schedule

```
                    LEARNING RATE SCHEDULE

 LR
 1.0 |         +-----------------------+
     |        /                         \
     |       /                           \
     |      /                             \
     |     /                               \
 0.0 +----+---+-------------------------+---+-----> steps
     0   warmup                      warmdown  end
         (0%)                         (50%)

 Default: No warmup, 50% warmdown to 0

 Code:
 def get_lr_multiplier(it):
     if it < warmup_iters:
         return (it + 1) / warmup_iters          # Linear warmup
     elif it <= num_iterations - warmdown_iters:
         return 1.0                               # Full LR
     else:
         progress = (num_iterations - it) / warmdown_iters
         return progress * 1.0 + (1-progress) * final_lr_frac  # Linear decay
```

### 10.3 Gradient Accumulation

```
                    GRADIENT ACCUMULATION

 total_batch_size = 524288 tokens
 device_batch_size = 32 sequences
 sequence_length = 2048 tokens

 Tokens per micro-batch = 32 * 2048 = 65536

 With 8 GPUs: world_tokens = 65536 * 8 = 524288

 grad_accum_steps = 524288 / 524288 = 1 (no accumulation needed!)

 On single GPU:
 grad_accum_steps = 524288 / 65536 = 8 (need 8 micro-batches)

 +--------+--------+--------+--------+--------+--------+--------+--------+
 | micro  | micro  | micro  | micro  | micro  | micro  | micro  | micro  |
 |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    |
 +--------+--------+--------+--------+--------+--------+--------+--------+
 |<---------------- gradient accumulation ----------------->|  optimizer
                                                               step()
```

---

## 11. Complete Data Flow

### 11.1 Forward Pass (Training)

```
                    COMPLETE FORWARD PASS

 +-------------------------------------------------------------------+
 | Input: token_ids [B=32, T=2048]                                   |
 +-------------------------------------------------------------------+
                              |
                              v
 +-------------------------------------------------------------------+
 | Token Embedding [B, T] -> [B, T, 768]                             |
 | wte.weight: [32768, 768]                                          |
 +-------------------------------------------------------------------+
                              |
                              v
 +-------------------------------------------------------------------+
 | RMSNorm (no params)                                               |
 | x = x / sqrt(mean(x^2) + eps)                                     |
 +-------------------------------------------------------------------+
                              |
                              v
 +-------------------------------------------------------------------+
 |                     TRANSFORMER BLOCK x12                         |
 |                                                                   |
 | For each block i:                                                 |
 |   x = resid_lambdas[i] * x + x0_lambdas[i] * x0                  |
 |                                                                   |
 |   ATTENTION:                                                      |
 |   - Q = c_q(norm(x))     [B,T,768] -> [B,T,6,128]                |
 |   - K = c_k(norm(x))     [B,T,768] -> [B,T,6,128]                |
 |   - V = c_v(norm(x))     [B,T,768] -> [B,T,6,128]                |
 |   - Apply RoPE to Q, K                                           |
 |   - QK-Norm                                                       |
 |   - FlashAttention(Q, K, V) -> [B,T,6,128]                       |
 |   - Reshape and project: c_proj -> [B,T,768]                     |
 |   - x = x + attn_output                                          |
 |                                                                   |
 |   MLP:                                                            |
 |   - h = c_fc(norm(x))    [B,T,768] -> [B,T,3072]                 |
 |   - h = relu(h)^2                                                |
 |   - h = c_proj(h)        [B,T,3072] -> [B,T,768]                 |
 |   - x = x + h                                                    |
 +-------------------------------------------------------------------+
                              |
                              v
 +-------------------------------------------------------------------+
 | Final RMSNorm                                                     |
 +-------------------------------------------------------------------+
                              |
                              v
 +-------------------------------------------------------------------+
 | LM Head: [B, T, 768] -> [B, T, 32768]                            |
 | + Logit softcap: 15 * tanh(logits / 15)                          |
 +-------------------------------------------------------------------+
                              |
                              v
 +-------------------------------------------------------------------+
 | Cross-Entropy Loss with targets                                   |
 | loss = -log(softmax(logits)[target])                             |
 +-------------------------------------------------------------------+
                              |
                              v
                    loss.backward()
                              |
                              v
 +-------------------------------------------------------------------+
 | Gradients flow backward through all layers                        |
 | optimizer.step() updates all parameters                           |
 +-------------------------------------------------------------------+
```

### 11.2 Parameter Count Breakdown

```
                    PARAMETER BREAKDOWN (depth=12)

 +------------------+----------------+------------------+
 | Component        | Shape          | Parameters       |
 +------------------+----------------+------------------+
 | wte (embedding)  | [32768, 768]   | 25,165,824      |
 +------------------+----------------+------------------+
 | Per Block (x12):                                     |
 |   c_q            | [768, 768]     | 589,824         |
 |   c_k            | [768, 768]     | 589,824         |
 |   c_v            | [768, 768]     | 589,824         |
 |   c_proj         | [768, 768]     | 589,824         |
 |   c_fc           | [768, 3072]    | 2,359,296       |
 |   c_proj (mlp)   | [3072, 768]    | 2,359,296       |
 |   ve_gate (alt)  | [32, 6]        | 192 (some layers)|
 | Block total      |                | ~7,077,888      |
 +------------------+----------------+------------------+
 | All blocks       | 12 blocks      | ~85,000,000     |
 +------------------+----------------+------------------+
 | value_embeds     | 6 x [32768,768]| 150,994,944     |
 +------------------+----------------+------------------+
 | lm_head          | [768, 32768]   | 25,165,824      |
 +------------------+----------------+------------------+
 | resid_lambdas    | [12]           | 12              |
 | x0_lambdas       | [12]           | 12              |
 +------------------+----------------+------------------+
 | TOTAL            |                | ~124M params    |
 +------------------+----------------+------------------+

 For depth=24 (GPT-2 scale): ~350M parameters
```

---

## Quick Reference: File to Concept Mapping

```
 +------------------------+-------------------------------------------+
 | File                   | Key Concepts                              |
 +------------------------+-------------------------------------------+
 | nanochat/gpt.py        | GPT, GPTConfig, Block, CausalSelfAttention|
 |                        | MLP, RoPE, Value Embeddings               |
 +------------------------+-------------------------------------------+
 | nanochat/engine.py     | Engine, KVCache, generate(), Calculator   |
 +------------------------+-------------------------------------------+
 | nanochat/tokenizer.py  | RustBPETokenizer, render_conversation     |
 +------------------------+-------------------------------------------+
 | nanochat/dataloader.py | Best-fit packing, BOS alignment           |
 +------------------------+-------------------------------------------+
 | nanochat/optim.py      | MuonAdamW, Polar Express, AdamW           |
 +------------------------+-------------------------------------------+
 | nanochat/flash_attention.py | FA3/SDPA switching, sliding window   |
 +------------------------+-------------------------------------------+
 | scripts/base_train.py  | Training loop, LR schedule, logging       |
 +------------------------+-------------------------------------------+
```

---

## Summary

nanochat implements a modern GPT architecture with several key innovations:

1. **Rotary Position Embeddings (RoPE)** - Relative position encoding via rotation
2. **QK-Normalization** - Stabilizes attention scores during training
3. **ReLU^2 Activation** - Sparser than GELU, helps training
4. **Sliding Window Attention** - Reduces memory while preserving quality
5. **Value Embeddings** - ResFormer-style enhancement on alternating layers
6. **MuonAdamW Optimizer** - Orthogonalization for matrix parameters
7. **Best-Fit Document Packing** - Efficient training data utilization
8. **Flash Attention 3** - Hardware-optimized attention on Hopper GPUs

The codebase is intentionally minimal (~2000 lines of core code) while implementing a complete LLM training pipeline that can produce GPT-2 quality models for ~$73 on 8xH100 GPUs.
