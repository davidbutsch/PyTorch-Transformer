import math
import torch
import torch.nn as nn
from config import config


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.d_k = config["d_model"] // config["h"]

        # Initialize Q, K, V transform layers

        # T: R^d_model -> R^d_model (applied broadcast over (batch, seq_len))
        # by T(x) = x*W^T + b where W is (out, in), W^T is (in, out), x is (batch, seq_len, d_model)
        self.Q = nn.Linear(
            in_features=config["d_model"], out_features=config["d_model"]
        )
        self.K = nn.Linear(
            in_features=config["d_model"], out_features=config["d_model"]
        )
        self.V = nn.Linear(
            in_features=config["d_model"], out_features=config["d_model"]
        )

        # Initialize output layer

        # T: R^d_model -> R^d_model
        self.out_layer = nn.Linear(
            in_features=config["d_model"], out_features=config["d_model"]
        )

    # Input embeddings: (batch, seq_len, d_model)
    def forward(
        self,
        embeddings: torch.Tensor,
    ):

        batch_size = embeddings.shape[0]

        # Build causal mask
        seq_len = embeddings.shape[1]

        # Mask filteres all upper contents (which are True), leaves just lower triangular (which are False)
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,  # dtype=torch.bool -> 0: False, 1: True
                requires_grad=False,
                device=config["device"],
            ),
            diagonal=1,  # Diagonal=1 omits the diagonal in triu matrix
        )

        # Compute h attention head outputs representations of input embeddings (but in single linear operation)

        q: torch.Tensor = self.Q(embeddings)  # (batch, seq_len, d_model)
        k: torch.Tensor = self.K(embeddings)  # ...
        v: torch.Tensor = self.V(embeddings)  # ...

        # Reshape query, key, and value representations to expose h dimension
        q = q.reshape((batch_size, seq_len, config["h"], self.d_k))
        k = k.reshape((batch_size, seq_len, config["h"], self.d_k))
        v = v.reshape((batch_size, seq_len, config["h"], self.d_k))

        # Transpose representations so that h is a batch dimension
        q = q.transpose(dim0=2, dim1=1)  # (batch_size, h, seq_len, d_k)
        k = k.transpose(dim0=2, dim1=1)  # ...
        v = v.transpose(dim0=2, dim1=1)  # ...

        # Compute attention scores (dot product similarity between query and key representations)
        # Scale scores by sqrt(d_k)^-1 for numerical stability

        k_t = k.transpose(dim0=2, dim1=3)  # Swap dimensions: seq_len (2), d_k (3)

        # So...
        # q:    (batch_size, h, seq_len, d_k)
        # k^T:  (batch_size, h, d_k, seq_len)

        # Performs batched matmul (ignoring "batch" dimensions which is all dimensions besides the last 2)
        # q: (batch_size, h, seq_len, d_k) @ k_t: (batch_size, h, d_k, seq_len) -> (seq_len, d_k) @ (d_k, seq_len) -> (seq_len, seq_len)
        scores = q @ k_t / math.sqrt(self.d_k)  # (batch_size, h, seq_len, seq_len)

        # Apply causal mask -> upper triangular entries set to -1e9 (0 after softmax)
        scores = scores.masked_fill(causal_mask, -1e9)

        # Compute attention scores row-wise
        weights = torch.softmax(
            input=scores,
            dim=3,  # We iterate over the column dimension (batch_size, h, seq_len, **seq_len**) to compute softmax
        )  # (batch_size, h, seq_len, seq_len) in [0, 1) where sum of each row dimension slice (batch_size, h, **seq_len**, seq_len) is 1

        # Compute context-enhanced embeddings (value representation as a weighted sum of the similarity distribution)
        c_embeddings = (
            weights @ v
        )  # Batch matmul -> weights: (batch_size, h, seq_len, seq_len) @ v: (batch_size, h, seq_len, d_k) -> (seq_len, seq_len) @ (seq_len, d_k) -> (seq_len, d_k)
        # c_embeddings: (batch_size, h, seq_len, d_k)

        # Move the h dimension back next to d_k so that we can combine these dimensions back to a single d_model dimension
        c_embeddings = c_embeddings.transpose(
            dim0=1, dim1=2
        )  # (batch_size, seq_len, h, d_k)

        # Combine dimensions h (2), d_k (3)
        # (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        c_embeddings = c_embeddings.reshape((batch_size, seq_len, config["d_model"]))

        # Make c_embeddings contiguous in memory (so that reshape doesn't produce wrong results)
        c_embeddings = c_embeddings.contiguous()

        # Compute output: concat @ W_o
        out = self.out_layer(c_embeddings)

        return out
