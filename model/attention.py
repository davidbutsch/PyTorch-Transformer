import math
import torch
import torch.nn as nn


class MaskedMultiHeadAttention(nn.Module):
    causal_mask: torch.Tensor

    def __init__(self, d_model: int, h: int, max_seq_len: int) -> None:
        super().__init__()

        self.h   = h
        self.d_k = d_model // h

        # T: R^d_model -> R^d_model (applied broadcast over (batch, seq_len))
        # by T(x) = x*W^T + b where W is (out, in), W^T is (in, out), x is (batch, seq_len, d_model)
        self.Q = nn.Linear(in_features=d_model, out_features=d_model)
        self.K = nn.Linear(in_features=d_model, out_features=d_model)
        self.V = nn.Linear(in_features=d_model, out_features=d_model)

        # T: R^d_model -> R^d_model
        self.out_layer = nn.Linear(in_features=d_model, out_features=d_model)

        # Pre-built causal mask registered as a buffer so it lives on the same
        # device as the model and is never re-allocated during the forward pass.
        # Mask filters all upper contents (which are True), leaves just lower triangular (which are False)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1),
        )

    # Input embeddings: (batch, seq_len, d_model)
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = embeddings.shape[0]
        seq_len    = embeddings.shape[1]

        # Compute h attention head representations in a single linear operation
        q: torch.Tensor = self.Q(embeddings)  # (batch, seq_len, d_model)
        k: torch.Tensor = self.K(embeddings)
        v: torch.Tensor = self.V(embeddings)

        # Reshape to expose the h dimension, then transpose so h is a batch dim
        q = q.reshape((batch_size, seq_len, self.h, self.d_k))
        k = k.reshape((batch_size, seq_len, self.h, self.d_k))
        v = v.reshape((batch_size, seq_len, self.h, self.d_k))

        q = q.transpose(dim0=2, dim1=1)  # (batch_size, h, seq_len, d_k)
        k = k.transpose(dim0=2, dim1=1)
        v = v.transpose(dim0=2, dim1=1)

        # Scaled dot-product attention scores
        # q @ k^T: (b, h, seq_len, d_k) @ (b, h, d_k, seq_len) -> (b, h, seq_len, seq_len)
        scores = q @ k.transpose(dim0=2, dim1=3) / math.sqrt(self.d_k)

        # Apply sliced pre-built causal mask -> upper triangular entries set to -inf (0 after softmax)
        scores = scores.masked_fill(self.causal_mask[:seq_len, :seq_len], float("-inf"))

        # Compute attention weights row-wise
        weights = torch.softmax(input=scores, dim=3)
        # (batch_size, h, seq_len, seq_len) in [0, 1) where sum of each row is 1

        # Weighted sum of value representations
        c_embeddings = weights @ v  # (batch_size, h, seq_len, d_k)

        # Move the h dimension back next to d_k so that we can combine these dimensions back to a single d_model dimension
        c_embeddings = c_embeddings.transpose(dim0=1, dim1=2)  # (batch_size, seq_len, h, d_k)

        # (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        c_embeddings = c_embeddings.reshape((batch_size, seq_len, self.h * self.d_k))
        c_embeddings = c_embeddings.contiguous()

        return self.out_layer(c_embeddings)
