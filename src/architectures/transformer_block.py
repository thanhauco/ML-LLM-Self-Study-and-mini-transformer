import torch
import torch.nn as nn
from src.architectures.attention import SelfAttention

class TransformerBlock(nn.Module):
    """
    A single Transformer layer (Pre-norm variant).
    """
    def __init__(self, d_model, ff_hidden, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention sub-layer
        attn_out, weights = self.attention(self.norm1(x), mask=mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward sub-layer
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x, weights

if __name__ == "__main__":
    d_model = 32
    ff_hidden = 128
    block = TransformerBlock(d_model, ff_hidden)
    
    x = torch.randn(2, 10, d_model)
    out, weights = block(x)
    print(f"Block output shape: {out.shape}")
