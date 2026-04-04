import math
import torch
import torch.nn as nn
from config import config


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.d_k = config["d_model"] // config["h"]

        # Initialize h Q_i, K_i, V_i transform layers

        # T: R^d_model -> R^d_model (applied broadcast over (batch, seq_len))
        # by T(x) = x*W^T + b where W is (out, in), W^T is (in, out), x is (batch, seq_len, d_model)
        self.q_layers = nn.ModuleList(
            [
                nn.Linear(in_features=config["d_model"], out_features=self.d_k)
                for _ in range(config["h"])
            ]
        )
        self.k_layers = nn.ModuleList(
            [
                nn.Linear(in_features=config["d_model"], out_features=self.d_k)
                for _ in range(config["h"])
            ]
        )
        self.v_layers = nn.ModuleList(
            [
                nn.Linear(in_features=config["d_model"], out_features=self.d_k)
                for _ in range(config["h"])
            ]
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

        # Compute h attention head outputs representations of input embeddings

        heads: list[torch.Tensor] = []  # [ (batch, seq_len, d_k) ]

        for i in range(config["h"]):
            q_i: torch.Tensor = self.q_layers[i](embeddings)  # (batch, seq_len, d_k)
            k_i: torch.Tensor = self.k_layers[i](embeddings)  # ...
            v_i: torch.Tensor = self.v_layers[i](embeddings)  # ...

            # Compute attention scores (dot product similarity between query and key representations)
            # Scale scores by sqrt(d_k)^-1 for numerical stability

            k_i_transpose = k_i.transpose(
                dim0=1, dim1=2
            )  # Swap dimensions: seq_len (1), d_k (2)

            # So...
            # Q_i:    (batch, seq_len, d_k)
            # K_i^T:  (batch, d_k, seq_len)

            # Performs batched matmul (ignoring "batch" dimensions which is all dimensions besides the last 2)
            # q:(batch, seq_len, d_k) @ k_t:(batch, d_k, seq_len) -> (seq_len, d_k) @ (d_k, seq_len)
            scores = q_i @ k_i_transpose / math.sqrt(self.d_k)

            # Apply causal mask -> upper triangular entries set to -1e9 (0 after softmax)
            scores = scores.masked_fill(causal_mask, -1e9)

            # Compute attention scores row-wise
            weights = torch.softmax(
                input=scores,
                dim=2,  # We iterate over the column dimension (batch, seq_len, **seq_len**) to compute softmax
            )  # (batch, seq_len, seq_len) in [0, 1) where sum of each row dimension slice (batch, **seq_len**, seq_len) is 1

            # Compute context-enhanced embeddings (value representation as a weighted sum of the similarity distribution)
            c_embeddings = weights @ v_i  # Batch matmul -> (batch, seq_len, d_k)

            # Push attention output to heads list
            heads.append(c_embeddings)

        # Concat heads
        concat = torch.cat(
            heads,
            dim=2,
        )  # (batch, seq_len, d_k*h) -> (batch, seq_len, d_model)

        # Compute output: concat @ W_o
        out = self.out_layer(concat)

        return out
