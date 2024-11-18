import torch
import torch.optim as optim
from projects.mini_transformer.model import MiniTransformer
from projects.mini_transformer.utils import CharTokenizer, get_batch

# Hyperparameters
batch_size = 32
block_size = 64
max_iters = 500
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embd = 128
n_head = 4
n_layer = 3
dropout = 0.1

def train():
    # Use a tiny snippet of text for demonstration
    text = "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'."
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size
    
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    model = MiniTransformer(
        vocab_size=vocab_size, 
        d_model=n_embd, 
        n_heads=n_head, 
        n_layers=n_layer, 
        ff_hidden=4*n_embd, 
        max_seq_len=block_size,
        dropout=dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Training on {device} with {sum(p.numel() for p in model.parameters())} parameters")

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            model.eval()
            with torch.no_state_dict():
                xb, yb = get_batch(train_data, block_size, batch_size)
                _, train_loss = model(xb.to(device), yb.to(device))
                print(f"step {iter}: train loss {train_loss.item():.4f}")
            model.train()

        xb, yb = get_batch(train_data, block_size, batch_size)
        logits, loss = model(xb.to(device), yb.to(device))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save the model
    torch.save(model.state_dict(), 'mini_transformer.pth')
    print("Training complete. Model saved as mini_transformer.pth")

if __name__ == "__main__":
    # Note: block_size must be smaller than data length for get_batch to work simply
    # For this tiny demo, we'll adjust block_size if needed
    block_size = min(block_size, len("Machine learning is a field of inquiry") - 1)
    train()
