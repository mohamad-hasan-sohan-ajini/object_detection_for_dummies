import torch
from torch import nn


class ObjectQuery(nn.Module):
    """Learnable object query vectors."""

    def __init__(self, embedding_dim: int, num_queries: int) -> None:
        super().__init__()

        self.num_queries = num_queries
        self.embedding_dim = embedding_dim
        self.query_vectors = nn.Parameter(torch.empty(num_queries, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_vectors, mean=0.0, std=0.01)

    @property
    def vectors(self) -> torch.Tensor:
        return self.query_vectors

    def forward(self, batch_size) -> torch.Tensor:
        """Return query vectors as ``[num_queries, dim]`` or ``[batch, num_queries, dim]``."""
        if batch_size is None:
            return self.vectors

        return self.vectors.unsqueeze(0).expand(batch_size, -1, -1)
