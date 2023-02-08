"""Notebook for preprocessing of input text."""
# pylint: disable=import-error,redefined-outer-name,unnecessary-lambda-assignment,invalid-name

import torch
from torch import nn
from torch.nn import functional as F


# independent sequences processed in parallel
BATCH_SIZE = 32
# maximum context length for predictions
BLOCK_SIZE = 8
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 200


# pylint: disable=too-few-public-methods
class Head(nn.Module):
    """Self-attention head."""

    def __init__(self, head_size, n_embed=32):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        """Forward pass for self-attention head."""
        _, time_dim, channels = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * channels**-0.5
        wei = wei.masked_fill(self.tril[:time_dim, :time_dim] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        # weighted aggregation of values
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multi-headed attention module."""

    def __init__(self, num_heads, head_size, n_embed=32):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        """Forward pass for multi-headed attention."""
        # concatenate outputs by channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """A single linear layer followed by an activation function."""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), nn.ReLU(), nn.Linear(4 * n_embed, n_embed)
        )

    def forward(self, x):
        """Forward pass in feedforward layer."""
        return self.net(x)


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        # layer normalization
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        """Forward pass for transformer block."""
        # communication
        x = x + self.sa(self.ln1(x))
        # computation
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Simple general purpose transformer."""

    def __init__(self, vocab_size, n_embed=32):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, n_embed)
        # language model head
        self.lm_head = nn.Linear(n_embed, vocab_size)
        # multiple transformer blocks
        self.blocks = nn.Sequential(
            Block(n_embed, num_heads=4),
            Block(n_embed, num_heads=4),
            Block(n_embed, num_heads=4),
        )

    def forward(self, idx, targets=None):
        """Forward pass of network."""
        # returns (Batch, Time, Channels)
        batch, time = idx.shape

        token_embeddings = self.token_embedding_table(idx)
        position_embeddings = self.position_embedding_table(
            torch.arange(time, device=DEVICE)
        )
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            batch, time, channels = logits.shape
            logits = logits.view(batch * time, channels)
            targets = targets.view(batch * time)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate new tokens."""
        for _ in range(max_new_tokens):
            # crop context to prevent embedding overflow
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            # get predictions only for last character (latest time)
            logits = logits[:, -1, :]
            # convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # get 1 sample from multinomial distribution defined by probabilities
            idx_next = torch.multinomial(probs, num_samples=1)
            # append the result
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    # read file
    with open("input.txt", "r", encoding="utf-8") as file:
        text = file.read()

    print(f"Dataset length (in characters): {len(text)}")
    # preview data
    print(f"Data preview: {text[:1000]}")

    # get characters in text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Characters: {''.join(chars)}\nVocabulary size: {vocab_size}")

    # create basic encoder and decoder
    # this is a very simple encoder/decoder, some advanced, popular ones include:
    # - OpenAI's TikToken: https://github.com/openai/tiktoken
    # - Google's Sentencepiece: https://github.com/google/sentencepiece
    stoi = {c: i for i, c in enumerate(chars)}
    itos = dict(enumerate(chars))
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda v: "".join([itos[i] for i in v])

    # change data into torch tensor
    data = torch.tensor(encode(text), dtype=torch.long)
    # train/test split
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    # showcase the inputs and targets of a transformer
    # inputs = train_data[:BLOCK_SIZE]
    # targets = train_data[1 : BLOCK_SIZE + 1]
    # for t in range(BLOCK_SIZE):
    #     context = inputs[: t + 1]
    #     target = targets[t]
    #     print(f"For input {context} the target is {target}")

    def get_batch(split):
        """Structure data into a batch."""
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
        x, y = x.to(DEVICE), y.to(DEVICE)
        return x, y

    model = TinyGPT(vocab_size)
    model.to(DEVICE)
    logits, loss = model(*get_batch("train"))

    # Print text generated by model
    print(
        decode(
            model.generate(
                torch.zeros((1, 1), dtype=torch.long, device=DEVICE), max_new_tokens=100
            )[0].tolist()
        )
    )

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    @torch.no_grad()
    def estimate_loss():
        """Estimate loss of model."""
        result = {}
        model.eval()
        for split in ("train", "val"):
            losses = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            result[split] = losses.mean()
        model.train()
        return result

    for step in range(MAX_ITERS):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss()
            print(
                f"Step {step}: Train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        # get all gradients for parameters
        loss.backward()
        # update the gradients
        optimizer.step()

    # print model results after training
    print(
        decode(
            model.generate(
                torch.zeros((1, 1), dtype=torch.long, device=DEVICE),
                max_new_tokens=1000,
            )[0].tolist()
        )
    )
