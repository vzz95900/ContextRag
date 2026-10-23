"""Adaptive Neural Weight Predictor.

A lightweight neural module that predicts the multi-objective weights (alpha, beta, gamma)
dynamically based on the query representation.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveWeightPredictor(nn.Module):
    """
    A small Multi-Layer Perceptron that takes a query embedding
    and predicts optimal weights for Relevance, Coverage, and Support.
    """
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3)
        
        # Initialize weights such that the default output is close to the balanced 
        # configuration (alpha=0.5, beta=0.3, gamma=0.2) before any training occurs.
        # This provides a strong prior-informed starting point.
        nn.init.constant_(self.fc2.bias, 0.0)
        self.fc2.bias.data = torch.tensor([0.5, 0.3, 0.2])
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.01)

    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_embedding: Tensor of shape (batch_size, embedding_dim)
            
        Returns:
            Tensor of shape (batch_size, 3) representing (alpha, beta, gamma) summing to 1.
        """
        x = F.relu(self.fc1(query_embedding))
        logits = self.fc2(x)
        return F.softmax(logits, dim=-1)

# Global singleton for inference
_predictor_instance = None

def get_adaptive_weights(query_emb_list: list[float]) -> Tuple[float, float, float]:
    """
    Convenience function to compute adaptive weights for a single query embedding.
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = AdaptiveWeightPredictor(embedding_dim=len(query_emb_list))
        _predictor_instance.eval()
        
    with torch.no_grad():
        q_tensor = torch.tensor(query_emb_list, dtype=torch.float32).unsqueeze(0)
        weights = _predictor_instance(q_tensor).squeeze(0)
        
    alpha, beta, gamma = weights.tolist()
    return alpha, beta, gamma
