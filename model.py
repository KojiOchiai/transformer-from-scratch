import math

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq = seq
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq, d_model)
        pe = torch.zeros(seq, d_model)

        # Create a vector of shape (seq)
        position = torch.arange(0, seq, dtype=torch.float).unsqueeze(1)  # (seq, 1)

        # Create a vector of shape (d_model)
        # For numerical stability we will use log space because e^{ln(n)} = n.
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model / 2)

        # Apply sine to even indeices
        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # sin(position * (10000 ^ (2i / d_model)))

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # cos(position * (10000 ^ (2i / d_model)))

        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, seq, d_model)

        # Register the positional encoding as a buffer
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.pe, torch.Tensor):
            pe = self.pe
        else:
            raise ValueError("Positional encoding is not a tensor")
        x = x + pe[:, : x.shape[1], :].requires_grad_(False)  # (batch, seq, d_model)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.features = features
        self.eps = eps
        self.alpha = nn.Parameter(
            torch.ones(features)
        )  # alpha (multiplicative) is a learnable parameter
        self.bias = nn.Parameter(
            torch.zeros(features)
        )  # bias (additive) is a learnable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)  # (batch, seq, 1)
        # eps is to prevent dividing by zero or when sted is verry small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.droptou = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq, d_model) -> (batch, seq, d_ff) -> (batch, set, d_model)
        return self.linear_2(self.droptou(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        self.d_k = d_model // h
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.droptout = nn.Dropout(dropout)

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        dropout: nn.Dropout | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = query.shape[-1]
        # Just appply the formula from the paper
        # (batch, h, seq, d_k) --> (batch, h, seq, seq)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a verry low value (indicating -inf) to the positions where mask is True
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(
            attention_scores, dim=-1
        )  # (batch, h, seq, seq), -inf will be zero
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq, seq) --> (batch, h, seq, d_k)
        # return attention scores which can be userd for visualization
        return (attention_scores @ value), attention_scores

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.w_q(q)  # (batch, seq, d_model) --> (batch, seq, d_model)
        key = self.w_k(k)  # (batch, seq, d_model) --> (batch, seq, d_model)
        value = self.w_v(v)  # (batch, seq, d_model) --> (batch, seq, d_model)

        # (batch, seq, d_model) --> (batch, seq, h, d_k) --> (batch, h, seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        # Calculate attention
        x, self.atention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.droptout
        )

        # Combine all the heads together
        # (batch, h, seq, d_k) --> (batch, seq, h, d_k) --> (batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq, d_model) --> (batch, seq, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(
        self, x: torch.Tensor, sublayer: MultiHeadAttentionBlock | FeedForwardBlock
    ) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


if __name__ == "__main__":
    d_model = 512
    seq = 10
    dropout = 0.1

    input_embedding = InputEmbedding(d_model, 100)
    positional_encoding = PositionalEncoding(d_model, seq, dropout)
    print(positional_encoding.pe)
    x = torch.randint(0, 100, (1, seq))
    print(x)
    x = input_embedding(x)
    print(x)
    x = positional_encoding(x)
    print(x)
