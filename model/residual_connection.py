import torch
import torch.nn as nn


class ResidualConnection(nn.Module):
    """Applies pre-norm residual connections across sequential sub-layers.

    For each sub-layer: x = x + SubLayer_i(LayerNorm_i(x))
    """

    def __init__(self, sub_layers: list[nn.Module], d_model: int) -> None:
        super().__init__()
        self.sub_layers = nn.ModuleList(sub_layers)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in sub_layers])

    def forward(self, c_embeddings: torch.Tensor) -> torch.Tensor:
        for sub_layer, layer_norm in zip(self.sub_layers, self.layer_norms):
            c_embeddings = c_embeddings + sub_layer(layer_norm(c_embeddings))
        return c_embeddings
