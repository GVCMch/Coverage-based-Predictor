import torch
import torch.nn as nn


class CriterionAwareGating(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )
        nn.init.constant_(self.gate_net[-1].bias, 1.0)

    def forward(self, criterion_features, model_emb):
        C = criterion_features.shape[1]
        model_emb_expanded = model_emb.expand(-1, C, -1)
        gate_input = torch.cat([criterion_features, model_emb_expanded], dim=-1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        return criterion_features * gate

    def get_gate_weights(self, criterion_features, model_emb):
        C = criterion_features.shape[1]
        model_emb_expanded = model_emb.expand(-1, C, -1)
        gate_input = torch.cat([criterion_features, model_emb_expanded], dim=-1)
        return torch.sigmoid(self.gate_net(gate_input)).squeeze(-1)
