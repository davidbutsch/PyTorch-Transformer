import torch
import torch.nn as nn
from config import config


class ResidualConnection(nn.Module):
    # Encompasses sequential sub-layers, normalizes and residually connects each output
    def __init__(self, sub_layers: list[nn.Module]) -> None:
        super().__init__()

        # Initialize modules into module list (for pytorch param tracking)
        self.sub_layers = nn.ModuleList(sub_layers)

        # Initialize 1 normalization layer per sublayer
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(config["d_model"]) for _ in sub_layers]
        )

    # Input contextual embeddings: (batch, seq_len, d_model)
    def forward(self, c_embeddings: torch.Tensor):

        # Apply pre-normalization: x = x + SubLayer_i(LayerNorm_i(x))
        for sub_layer, layer_norm in zip(self.sub_layers, self.layer_norms):
            c_embeddings = c_embeddings + sub_layer(layer_norm(c_embeddings))

        return c_embeddings
