import torch
import torch.nn.functional as F
from projects.mini_transformer.model import MiniTransformer
from projects.mini_transformer.utils import CharTokenizer

def generate(model, tokenizer, prompt, max_new_tokens, block_size, device):
    model.eval()
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        # Crop context if it exceeds block_size
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
    return tokenizer.decode(idx[0].tolist())

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text = "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'."
    tokenizer = CharTokenizer(text)
    
    # Initialize model
    model = MiniTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=3,
        ff_hidden=512,
        max_seq_len=64
    ).to(device)
    
    # In a real scenario, you'd load weights here:
    # model.load_state_dict(torch.load('mini_transformer.pth', map_location=device))
    
    prompt = "Machine"
    generated_text = generate(model, tokenizer, prompt, 50, 64, device)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
