import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """
    A minimal implementation of Scaled Dot-Product Attention.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # W_q, W_k, W_v weight matrices
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, d_model)
        B, L, D = x.shape
        
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        
        # Compute scores: (B, L, D) @ (B, D, L) -> (B, L, L)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(D)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attention_weights = F.softmax(scores, dim=-1)
        
        # (B, L, L) @ (B, L, D) -> (B, L, D)
        out = torch.matmul(attention_weights, values)
        
        return out, attention_weights

if __name__ == "__main__":
    # Test Self-Attention
    d_model = 16
    batch_size = 2
    seq_len = 5
    
    x = torch.randn(batch_size, seq_len, d_model)
    attn = SelfAttention(d_model)
    
    output, weights = attn(x)
    print(f"Output shape: {output.shape}") # Expect (2, 5, 16)
    print(f"Weights shape: {weights.shape}") # Expect (2, 5, 5)
