import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GatedRouter(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256,
                 num_experts: int = 4, entropy_weight: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.entropy_weight = entropy_weight
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, encoder_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = encoder_hidden_states.mean(dim=1)
        pooled = pooled.to(next(self.network.parameters()).dtype)
        logits = self.network(pooled)
        weights = F.softmax(logits, dim=-1)
        return weights, logits

    def entropy_loss(self, weights: torch.Tensor) -> torch.Tensor:
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
        return -self.entropy_weight * entropy

    def routing_stats(self, weights: torch.Tensor) -> dict:
        assignments = weights.argmax(dim=-1)
        counts = torch.bincount(assignments, minlength=self.num_experts)
        load_balance = counts.float() / counts.sum()
        mean_entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
        return {
            "load_balance": load_balance.tolist(),
            "mean_entropy": mean_entropy.item(),
            "mean_max_weight": weights.max(dim=-1).values.mean().item(),
        }