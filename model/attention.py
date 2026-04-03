import math
import torch
import torch.nn as nn
from config import config


# NOT MULTIHEAD YET -> d_k == d_model
class MaskedAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.d_model = config["d_model"]
        self.d_k = config["d_model"] // config["h"]

        # T: R^d_model -> R^d_k (applied broadcast over (batch, seq_len))
        # by T(x) = x*W^T + b where W is (out, in), W^T is (in, out), x is (batch, seq_len, d_model)
        self.q_layer = nn.Linear(in_features=self.d_model, out_features=self.d_k)
        self.k_layer = nn.Linear(in_features=self.d_model, out_features=self.d_k)
        self.v_layer = nn.Linear(in_features=self.d_model, out_features=self.d_k)

    # Input embeddings: (batch, seq_len, d_model)
    def forward(
        self,
        embeddings: torch.Tensor,
    ):
        # Compute Q, K, V representation of input embeddings

        q: torch.Tensor = self.q_layer(embeddings)  # (batch, seq_len, d_k)
        k: torch.Tensor = self.k_layer(embeddings)  # ...
        v: torch.Tensor = self.v_layer(embeddings)  # ...

        # Compute attention scores (dot product similarity between query and key representations)
        # Scale scores by sqrt(d_k)^-1 for numerical stability

        k_t = k.transpose(dim0=1, dim1=2)  # Swap dimensions seq_len (1), d_k (2)
        # So...
        # Q:    (batch, seq_len, d_k)
        # K^T:  (batch, d_k, seq_len)

        # Performs batched matmul (ignoring "batch" dimensions which is all dimensions besides the last 2)
        # q:(batch, seq_len, d_k) @ k_t:(batch, d_k, seq_len) -> (seq_len, d_k) @ (d_k, seq_len)
        scores = q @ k_t / math.sqrt(self.d_k)  # (batch, seq_len, seq_len)

        # Build causal mask
        seq_len = scores.shape[1]

        # Upper triangular mask blocks all upper triangular entries
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                requires_grad=False,
                device=config["device"],
            ),
            diagonal=1,
        )

        # Apply causal mask -> upper triangular entries set to -1e9 (0 after softmax)
        scores = scores.masked_fill(causal_mask, -1e9)

        weights = torch.softmax(
            input=scores,
            dim=2,  # We iterate over the column dimension (batch, seq_len, **seq_len**) to compute softmax
        )  # (batch, seq_len, seq_len) in [0, 1) where sum of each row dimension slice (batch, **seq_len**, seq_len) is 1

        # Compute contextual embeddings (value token representations as a weighted sum of the similarity distribution)
        contextual_embeddings = weights @ v  # Batch matmul -> (batch, seq_len, d_model)

        return contextual_embeddings  # (batch, seq_len, d_k)
