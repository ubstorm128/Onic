# Model Architecture

## 1. Architecture Overview

```
┌──────────┐     ┌──────────────┐     ┌─────────┐
│ mRNA     │────▶│ Encoder_mRNA │────▶│         │
│ (3217)   │     │              │     │         │
└──────────┘     └──────────────┘     │         │     ┌───────────┐     ┌──────────┐
                                      │ Fusion  │────▶│ Classifier│────▶│ 4 classes│
┌──────────┐     ┌──────────────┐     │ Module  │     │           │     │          │
│ miRNA    │────▶│ Encoder_miRNA│────▶│ (Cross- │     └───────────┘     └──────────┘
│ (383)    │     │              │     │  Attn)  │
└──────────┘     └──────────────┘     │         │
                                      │         │
┌──────────┐     ┌──────────────┐     │         │
│ CNV      │────▶│ Encoder_CNV  │────▶│         │
│ (3105)   │     │              │     │         │
└──────────┘     └──────────────┘     │         │
                                      │         │
┌──────────┐     ┌──────────────┐     │         │
│ Methy    │────▶│ Encoder_Methy│────▶│         │
│ (3139)   │     │              │     └─────────┘
└──────────┘     └──────────────┘
```

## 2. Modality-Specific Encoders

Each encoder maps a modality's feature space to a shared representation dimension D.

```python
class ModalityEncoder(nn.Module):
    """
    Two-layer MLP encoder for a single modality.
    
    Input:  (batch_size, input_dim)     # e.g., (64, 3217) for mRNA
    Output: (batch_size, output_dim)    # e.g., (64, 128)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128, dropout: float = 0.3):
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
```

**Design choice:** Simple 2-layer MLP rather than deeper network, because:
- The feature space is already curated (not raw sequences/images)
- Tabular data rarely benefits from very deep networks
- Overfitting risk is high with 2,263 samples

## 3. Fusion Module

### 3.1 Self-Attention Fusion

```python
class CrossAttentionFusion(nn.Module):
    """
    Multi-head self-attention over modality embeddings.
    
    Input:  List of (batch_size, D) tensors, one per available modality
    Output: (batch_size, D) fused representation
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, num_modalities: int = 4):
        self.modality_embeddings = nn.Embedding(num_modalities, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
```

**Tensor shapes through fusion:**

| Step | Shape | Description |
|---|---|---|
| Encoder outputs | 4 × (B, D) | One per modality |
| Stack | (B, 4, D) | Modality sequence |
| + modality embeddings | (B, 4, D) | Add positional identity |
| Self-attention | (B, 4, D) | Cross-modality interaction |
| Mean pool | (B, D) | Aggregate to single vector |

### 3.2 Missing-Modality Masking

When modalities are absent:

```python
# Create attention mask: True = masked (excluded)
# E.g., if miRNA is missing: mask = [False, True, False, False]
attn_mask = torch.zeros(batch_size, num_modalities, dtype=torch.bool)
for absent_idx in absent_modality_indices:
    attn_mask[:, absent_idx] = True

# Replace absent encoder outputs with learned defaults
for absent_idx in absent_modality_indices:
    modality_tokens[:, absent_idx, :] = self.default_embeddings[absent_idx]

# Apply masked attention
output, attn_weights = self.attention(
    modality_tokens, modality_tokens, modality_tokens,
    key_padding_mask=attn_mask
)

# Pool only over present modalities
present_mask = ~attn_mask  # (B, 4)
output = (output * present_mask.unsqueeze(-1)).sum(dim=1) / present_mask.sum(dim=1, keepdim=True)
```

### 3.3 Learned Default Embeddings

```python
self.default_embeddings = nn.Parameter(torch.randn(num_modalities, embed_dim))
```

These are optimized during training and represent "no information available" for each modality.

## 4. Classification Head

```python
class Classifier(nn.Module):
    """
    Input:  (batch_size, D)      # D = 128
    Output: (batch_size, 4)      # 4 cancer classes
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, num_classes: int = 4):
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )
```

## 5. Full Model

```python
class MultimodalCancerClassifier(nn.Module):
    def __init__(self, modality_dims, hidden_dim=256, embed_dim=128, num_heads=4, num_classes=4):
        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(dim, hidden_dim, embed_dim)
            for name, dim in modality_dims.items()
        })
        self.fusion = CrossAttentionFusion(embed_dim, num_heads, len(modality_dims))
        self.classifier = Classifier(embed_dim, 64, num_classes)
    
    def forward(self, modality_data: dict, available_mask: dict):
        # modality_data: {"mRNA": tensor, "miRNA": tensor, ...}
        # available_mask: {"mRNA": True, "miRNA": False, ...}
        embeddings = []
        for name, encoder in self.encoders.items():
            if available_mask.get(name, True):
                embeddings.append(encoder(modality_data[name]))
            else:
                embeddings.append(None)  # Will use default
        
        fused = self.fusion(embeddings, available_mask)
        logits = self.classifier(fused)
        return logits
```

## 6. Parameter Count Estimate

| Component | Parameters |
|---|---|
| mRNA encoder | 3217×256 + 256 + 256×128 + 128 ≈ 856K |
| miRNA encoder | 383×256 + 256 + 256×128 + 128 ≈ 131K |
| CNV encoder | 3105×256 + 256 + 256×128 + 128 ≈ 828K |
| Methy encoder | 3139×256 + 256 + 256×128 + 128 ≈ 837K |
| Fusion (attention) | 3×128×128 + 128 ≈ 49K |
| Modality embeddings | 4×128 = 512 |
| Default embeddings | 4×128 = 512 |
| Classifier | 128×64 + 64 + 64×4 + 4 ≈ 8.5K |
| **Total** | **~2.7M** |

This is modest for a neural network and appropriate for ~2,263 training samples (ratio ~1:1 samples:parameters after considering the effective parameters in play).
