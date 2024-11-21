import torch

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

def get_batch(data, block_size, batch_size):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

if __name__ == "__main__":
    # Tiny synthetic data test
    text = "hello world"
    tokenizer = CharTokenizer(text)
    print(f"Vocab size: {tokenizer.vocab_size}")
    encoded = tokenizer.encode("hello")
    print(f"Encoded 'hello': {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")
