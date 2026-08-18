import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """ One single head of self-attention """

    def __init__(self, head_size, embed_size, block_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        # Perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention running in parallel """

    def __init__(self, num_heads, head_size, embed_size, block_size):
        super().__init__()
        # Create a list of independent Attention Heads
        self.heads = nn.ModuleList([Head(head_size, embed_size, block_size) for _ in range(num_heads)])
        # Projection layer to merge head outputs back into the main pipeline
        self.proj = nn.Linear(num_heads * head_size, embed_size)

    def forward(self, x):
        # Run every head on the input data and concatenate the results side-by-side
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class MultiHeadLM(nn.Module):
    def __init__(self, vocab_size, embed_size, block_size, num_heads):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)

        # Initialize our new Multi-Head Attention layer
        head_size = embed_size // num_heads
        self.sa_heads = MultiHeadAttention(num_heads, head_size, embed_size, block_size)

        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        # Pass data through the multi-head layer
        x = self.sa_heads(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
