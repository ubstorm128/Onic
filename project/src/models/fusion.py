"""
Proposed Multimodal Cross-Attention Fusion Architecture
======================================================
Implements:
1. Modality-specific Encoders (mRNA, miRNA, CNV, Methy)
2. Learned Modality-Type Embeddings
3. Learned Default Missing-Modality Tokens
4. Multi-Head Attention Fusion over modality tokens
5. Mask-aware Token Aggregation (Mean Pooling over available modalities)
6. Classification Head
7. Export of Attention Weights for Explainability
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    """
    Encodes raw features of a single modality into a shared latent space D.
    Architecture:
        Linear(input_dim, hidden_dim) -> BN -> ReLU -> Dropout
        Linear(hidden_dim, output_dim) -> BN -> ReLU -> Dropout
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
            
        layers.append(nn.Linear(hidden_dim, output_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
            
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, input_dim) -> (batch_size, output_dim)"""
        return self.encoder(x)


class MultimodalCrossAttentionClassifier(nn.Module):
    """
    Full Multimodal AI Architecture for Cancer Classification under Missing Modalities.
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        encoder_hidden_dim: int = 256,
        latent_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        num_classes: int = 4,
        encoder_dropout: float = 0.3,
        fusion_dropout: float = 0.1,
        classifier_dropout: float = 0.3,
        use_modality_type_embeddings: bool = True,
        use_learned_default_token: bool = True,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.modality_keys = list(modality_dims.keys())
        self.num_modalities = len(self.modality_keys)
        self.latent_dim = latent_dim
        self.use_modality_type_embeddings = use_modality_type_embeddings
        self.use_learned_default_token = use_learned_default_token
        
        # 1. Dedicated Encoders
        self.encoders = nn.ModuleDict({
            mod: ModalityEncoder(
                input_dim=dim,
                hidden_dim=encoder_hidden_dim,
                output_dim=latent_dim,
                dropout=encoder_dropout,
            )
            for mod, dim in modality_dims.items()
        })
        
        # 2. Modality Type Embeddings (learned token per modality index)
        if use_modality_type_embeddings:
            self.modality_type_embeddings = nn.Parameter(
                torch.randn(1, self.num_modalities, latent_dim) * 0.02
            )
        else:
            self.modality_type_embeddings = None
            
        # 3. Learned Default Tokens for Missing Modalities (one per modality)
        if use_learned_default_token:
            self.default_missing_tokens = nn.Parameter(
                torch.randn(self.num_modalities, latent_dim) * 0.02
            )
        else:
            self.default_missing_tokens = None
            
        # 4. Multi-Head Self-Attention Fusion Layer
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=fusion_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        
        # Feedforward sub-layer
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(dim_feedforward, latent_dim),
        )
        
        # 5. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        modality_tensors: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            modality_tensors: Dict mapping modality name to (batch_size, feature_dim)
            mask: Binary tensor of shape (batch_size, num_modalities) where 1=present, 0=absent.
            return_attention_weights: Whether to return the attention map (B, num_modalities, num_modalities)
            
        Returns:
            logits: (batch_size, num_classes)
            attention_weights (optional): (batch_size, num_modalities, num_modalities)
        """
        batch_size = next(iter(modality_tensors.values())).shape[0]
        device = next(iter(modality_tensors.values())).device
        
        if mask is None:
            mask = torch.ones(batch_size, self.num_modalities, device=device)
            
        # Step 1: Encode each present modality
        token_list = []
        for i, mod in enumerate(self.modality_keys):
            x_mod = modality_tensors[mod]
            z_mod = self.encoders[mod](x_mod)  # (batch_size, latent_dim)
            
            # If missing modality handling: replace absent samples with learned default token
            if self.use_learned_default_token and self.default_missing_tokens is not None:
                m_i = mask[:, i : i + 1]  # (batch_size, 1)
                default_token = self.default_missing_tokens[i : i + 1, :].expand(batch_size, -1)
                z_mod = m_i * z_mod + (1.0 - m_i) * default_token
            elif not self.use_learned_default_token:
                m_i = mask[:, i : i + 1]
                z_mod = m_i * z_mod
                
            token_list.append(z_mod.unsqueeze(1))  # (batch_size, 1, latent_dim)
            
        # Shape: (batch_size, num_modalities, latent_dim)
        tokens = torch.cat(token_list, dim=1)
        
        # Step 2: Add learned modality type embeddings
        if self.modality_type_embeddings is not None:
            tokens = tokens + self.modality_type_embeddings
            
        # Step 3: Multi-Head Self-Attention across modality tokens
        # Key padding mask: True indicates token should be ignored in attention
        # PyTorch MHA convention: key_padding_mask has shape (B, S), True = ignored
        key_padding_mask = (mask == 0.0)
        
        # Guard: if an entire sample has all modalities missing, do not mask all tokens (prevents NaN in softmax)
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
            
        attn_out, attn_weights = self.attention(
            query=tokens,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        
        # Residual + Norm
        tokens = self.norm1(tokens + attn_out)
        
        # FFN + Residual + Norm
        tokens = self.norm2(tokens + self.ffn(tokens))
        
        # Step 4: Mask-aware pooling (aggregate modality tokens into single patient representation)
        # Weight by presence mask so absent modalities don't distort the pooled vector
        mask_expanded = mask.unsqueeze(-1)  # (batch_size, num_modalities, 1)
        masked_tokens = tokens * mask_expanded
        mask_sum = mask_expanded.sum(dim=1).clamp(min=1.0)  # (batch_size, 1)
        h_fused = masked_tokens.sum(dim=1) / mask_sum  # (batch_size, latent_dim)
        
        # Step 5: Classification head
        logits = self.classifier(h_fused)
        
        if return_attention_weights:
            return logits, attn_weights
        return logits, None
