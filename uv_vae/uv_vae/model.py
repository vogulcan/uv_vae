from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class VAEConfig:
    categorical_cardinalities: dict[str, int]
    embedding_dims: dict[str, int]
    numeric_dim: int
    hidden_dims: list[int]
    latent_dim: int


class TabularVAE(nn.Module):
    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        self.categorical_names = list(config.categorical_cardinalities)
        self.numeric_dim = config.numeric_dim

        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(config.categorical_cardinalities[name], config.embedding_dims[name])
                for name in self.categorical_names
            }
        )

        encoder_layers: list[nn.Module] = []
        decoder_layers: list[nn.Module] = []
        input_dim = sum(config.embedding_dims.values()) + config.numeric_dim

        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(prev_dim, config.latent_dim)
        self.logvar = nn.Linear(prev_dim, config.latent_dim)

        prev_dim = config.latent_dim
        for hidden_dim in reversed(config.hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        self.decoder = nn.Sequential(*decoder_layers)
        self.numeric_head = nn.Linear(prev_dim, config.numeric_dim)
        self.categorical_heads = nn.ModuleDict(
            {
                name: nn.Linear(prev_dim, config.categorical_cardinalities[name])
                for name in self.categorical_names
            }
        )

    def encode(self, categorical_inputs: torch.Tensor, numeric_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pieces = []
        for index, name in enumerate(self.categorical_names):
            pieces.append(self.embeddings[name](categorical_inputs[:, index]))
        pieces.append(numeric_inputs)
        encoded = self.encoder(torch.cat(pieces, dim=1))
        return self.mu(encoded), self.logvar(encoded)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        decoded = self.decoder(latent)
        numeric_output = self.numeric_head(decoded)
        categorical_output = {
            name: head(decoded) for name, head in self.categorical_heads.items()
        }
        return numeric_output, categorical_output

    def forward(
        self,
        categorical_inputs: torch.Tensor,
        numeric_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(categorical_inputs, numeric_inputs)
        latent = self.reparameterize(mu, logvar)
        numeric_output, categorical_output = self.decode(latent)
        return numeric_output, categorical_output, mu, logvar
