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
        pe[:, 0::2] == torch.sin(
            position * div_term
        )  # sin(position * (10000 ^ (2i / d_model)))

        # Apply cosine to odd indices
        pe[:, 1::2] == torch.cos(
            position * div_term
        )  # cos(position * (10000 ^ (2i / d_model)))

        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, seq, d_model)

        # Register the positional encoding as a buffer
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.shape[1], :].requires_grad_(  # type: ignore
            False
        )  # (batch, seq, d_model)
        return self.dropout(x)
