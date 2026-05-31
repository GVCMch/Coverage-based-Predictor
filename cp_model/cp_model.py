import torch
import torch.nn as nn
from typing import List

from .cag import CriterionAwareGating


class MultiMetricMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        num_models: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.model_embed = nn.Embedding(num_models, 64)

        layers = []
        prev_dim = input_dim + 64

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features, model_ids):
        model_emb = self.model_embed(model_ids)
        x = torch.cat([features, model_emb], dim=1)
        logits = self.mlp(x)
        return logits.squeeze(-1)


class CoverageTransformer(nn.Module):

    def __init__(
        self,
        pca_dim: int = 256,
        num_criteria: int = 10,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_models: int = 3,
    ):
        super().__init__()
        self.pca_dim = pca_dim
        self.num_criteria = num_criteria
        self.d_model = d_model

        self.criterion_proj = nn.Linear(pca_dim, d_model)
        self.criterion_embed = nn.Embedding(num_criteria, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.model_embed = nn.Embedding(num_models, d_model)
        self.cag = CriterionAwareGating(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, features, model_ids, criterion_drop_prob=0.0, disable_cag=False):
        B = features.shape[0]

        features = features.view(B, self.num_criteria, self.pca_dim)

        if self.training and criterion_drop_prob > 0:
            keep_mask = torch.rand(B, self.num_criteria, 1, device=features.device) > criterion_drop_prob
            min_keep = self.num_criteria // 2
            for b in range(B):
                if keep_mask[b].sum() < min_keep:
                    dropped = (~keep_mask[b].squeeze(-1)).nonzero(as_tuple=True)[0]
                    restore = dropped[torch.randperm(len(dropped))[:min_keep - int(keep_mask[b].sum())]]
                    keep_mask[b, restore] = True
            features = features * keep_mask
            keep_ratio = keep_mask.float().mean(dim=1, keepdim=True).clamp(min=0.1)
            features = features / keep_ratio

        x = self.criterion_proj(features)

        criterion_ids = torch.arange(self.num_criteria, device=features.device)
        criterion_emb = self.criterion_embed(criterion_ids)
        x = x + criterion_emb.unsqueeze(0)

        model_emb = self.model_embed(model_ids).unsqueeze(1)

        if not disable_cag:
            x = self.cag(x, model_emb)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, model_emb, x], dim=1)
        x = self.transformer(x)

        cls_output = x[:, 0, :]
        logits = self.fc(cls_output)
        return logits.squeeze(-1)

    def get_cls_representation(self, features, model_ids):
        """Return h_CLS(x): the CLS token output used for LSCC clustering (paper Section 3.3)."""
        B = features.shape[0]
        features = features.view(B, self.num_criteria, self.pca_dim)
        x = self.criterion_proj(features)
        criterion_ids = torch.arange(self.num_criteria, device=features.device)
        criterion_emb = self.criterion_embed(criterion_ids)
        x = x + criterion_emb.unsqueeze(0)
        model_emb = self.model_embed(model_ids).unsqueeze(1)
        x = self.cag(x, model_emb)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, model_emb, x], dim=1)
        x = self.transformer(x)
        return x[:, 0, :]  # h_CLS ∈ R^{d_model}

    def get_gate_weights(self, features, model_ids):
        B = features.shape[0]
        features = features.view(B, self.num_criteria, self.pca_dim)
        x = self.criterion_proj(features)
        criterion_ids = torch.arange(self.num_criteria, device=features.device)
        criterion_emb = self.criterion_embed(criterion_ids)
        x = x + criterion_emb.unsqueeze(0)
        model_emb = self.model_embed(model_ids).unsqueeze(1)
        return self.cag.get_gate_weights(x, model_emb)
