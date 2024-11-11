import torch
import torch.nn as nn
from src.architectures.transformer_block import TransformerBlock

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, ff_hidden, max_seq_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_hidden, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embedding(idx) + self.pos_embedding[:, :T, :]
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0).to(idx.device)
        
        for block in self.blocks:
            x, _ = block(x, mask=mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)
            
        return logits, loss

if __name__ == "__main__":
    model = MiniTransformer(vocab_size=100, d_model=128, n_heads=4, n_layers=2, ff_hidden=512, max_seq_len=64)
    idx = torch.randint(0, 100, (2, 32))
    logits, _ = model(idx)
    print(f"Logits shape: {logits.shape}") # (2, 32, 100)
