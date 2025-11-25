import torch
import torch.nn as nn
import numpy as np
from simformer.model.tokenizer import Tokenizer
from simformer.model.graph import GraphInversion

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier features for encoding time."""
    def __init__(self, embedding_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        t_proj = t[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

class SimformerBlock(nn.Module):
    """A single block of the Simformer, i.e., a standard Transformer encoder layer."""
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Pre-norm
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_output)

        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)
        return x

class Simformer(nn.Module):
    """
    The core Simformer model.
    Integrates tokenizer, graph inversion, and a Transformer encoder.
    """
    def __init__(self, m_e, num_vars, embedding_dim=128, num_layers=6, num_heads=8, d_ff=512, dropout=0.1):
        super().__init__()
        self.num_vars = num_vars
        self.embedding_dim = embedding_dim
        self.d_model = embedding_dim * 4  # Concatenation of 4 embeddings

        self.tokenizer = Tokenizer(num_vars, embedding_dim)
        self.graph_inversion = GraphInversion(m_e)
        self.m_e_tensor = torch.from_numpy(m_e).bool()

        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embedding_dim),
            nn.Linear(embedding_dim, self.d_model)
        )

        self.transformer_blocks = nn.ModuleList(
            [SimformerBlock(self.d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(self.d_model)

    def forward(self, values, cond_mask, ids, metadata, t):
        """
        Forward pass for the Simformer.

        Args:
            values (torch.Tensor): Tensor of variable values. Shape (batch_size, num_vars).
            cond_mask (torch.Tensor): Boolean tensor indicating conditioned variables. Shape (batch_size, num_vars).
            ids (torch.Tensor): Tensor of variable identifiers. Shape (batch_size, num_vars).
            metadata (torch.Tensor): Tensor of variable metadata (e.g., param vs. data). Shape (batch_size, num_vars).
            t (torch.Tensor): Diffusion time step. Shape (batch_size,).

        Returns:
            torch.Tensor: The output sequence from the Transformer. Shape (batch_size, num_vars, d_model).
        """
        device = values.device
        batch_size = values.shape[0]

        # 1. Tokenize inputs
        x = self.tokenizer(values, cond_mask, ids, metadata)

        # 2. Compute time embedding
        time_embedding = self.time_embed(t)

        # 3. Compute attention mask (dynamically per sample)
        # This is computationally intensive as it's per-sample.
        # In a real high-performance scenario, one might look for ways to batch this.
        attention_masks = []
        for i in range(batch_size):
            m_c = cond_mask[i].cpu().numpy()
            h_mask = self.graph_inversion.compute_attention_mask(m_c)
            h_mask_tensor = torch.from_numpy(h_mask).bool().to(device)
            full_mask = self.m_e_tensor.to(device) | h_mask_tensor
            # Invert mask for nn.MultiheadAttention: True means "don't attend"
            attention_masks.append(~full_mask)

        attn_mask_batch = torch.stack(attention_masks, dim=0)

        # 4. Pass through Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attn_mask_batch)
            # Add time embedding after each block
            x = x + time_embedding.unsqueeze(1)

        # 5. Final normalization
        x = self.output_norm(x)

        return x
