"""Section 3.3: visual context encoder and bidirectional trajectory decoder."""
from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass
class ModelConfig:
    observed: int = 10
    future: int = 10
    feature_dim: int = 768
    dim: int = 256
    heads: int = 8
    ff_dim: int = 1024
    layers: int = 2
    dropout: float = 0.1
    use_video: bool = True
    use_pose: bool = True
    trajectory_only: bool = False
    attention: str = "bidirectional"


def temporal_embedding(length, dim):
    pos = torch.arange(length).float().unsqueeze(1)
    scale = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(pos * scale)
    pe[:, 1::2] = torch.cos(pos * scale[:dim // 2])
    return pe


class ScanpathTransformer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or ModelConfig()
        if c.observed < 1 or c.future < 1 or c.dim % c.heads:
            raise ValueError("Positive sequence lengths and dim divisible by heads required")
        if c.attention not in {"bidirectional", "causal", "partial"}:
            raise ValueError("Unknown attention mode")
        self.fixation = nn.Linear(4, c.dim)
        self.queries = nn.Parameter(torch.randn(c.future, c.dim) * 0.02)
        self.register_buffer("time", temporal_embedding(c.observed + c.future, c.dim))
        if not c.trajectory_only:
            self.dense = nn.Linear(c.feature_dim, c.dim)
            self.global_feature = nn.Linear(c.feature_dim, c.dim)
            self.pose = nn.Linear(7, c.dim)
            self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                c.dim, c.heads, c.ff_dim, c.dropout, batch_first=True), c.layers,
                enable_nested_tensor=False)
            self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
                c.dim, c.heads, c.ff_dim, c.dropout, batch_first=True), c.layers)
        else:
            self.decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                c.dim, c.heads, c.ff_dim, c.dropout, batch_first=True), c.layers,
                enable_nested_tensor=False)
        self.output = nn.Linear(c.dim, 4)
        # PyTorch clones layers with identical initial weights; initialize independently.
        for p in self.parameters():
            if p.ndim > 1 and p is not self.queries:
                nn.init.xavier_uniform_(p)

    def attention_mask(self, device):
        c = self.config
        if c.attention == "bidirectional":
            return None
        mask = torch.ones(c.observed + c.future, c.observed + c.future,
                          dtype=torch.bool, device=device).triu(1)
        if c.attention == "partial":
            mask[:c.observed, :c.observed] = False
        return mask

    def forward(self, observed, global_features=None, dense_features=None, poses=None):
        c = self.config
        if observed.ndim != 3 or observed.shape[1:] != (c.observed, 4):
            raise ValueError(f"observed must be [B,{c.observed},4]")
        b = observed.shape[0]
        past = self.fixation(observed) + self.time[:c.observed]
        # Paper adds temporal PE to observations; future queries are learned separately.
        trajectory = torch.cat([past, self.queries.unsqueeze(0).expand(b, -1, -1)], 1)
        mask = self.attention_mask(observed.device)
        if c.trajectory_only:
            decoded = self.decoder(trajectory, mask=mask)
        else:
            if dense_features is None or dense_features.ndim != 3 or dense_features.shape[0] != b or dense_features.shape[2] != c.feature_dim:
                raise ValueError("dense_features must be [B,P,feature_dim]")
            context = [self.dense(dense_features) + self.time[c.observed - 1]]
            if c.use_video:
                if global_features is None or global_features.shape != (b, c.observed - 1, c.feature_dim):
                    raise ValueError("global_features must contain exactly the preceding observed frames")
                context.insert(0, self.global_feature(global_features) + self.time[:c.observed - 1])
            if c.use_pose:
                if poses is None or poses.shape != (b, c.observed, 7):
                    raise ValueError("poses must be [B,observed,7]")
                context.append(self.pose(poses) + self.time[:c.observed])
            memory = self.encoder(torch.cat(context, 1))
            decoded = self.decoder(trajectory, memory, tgt_mask=mask)
        return self.output(decoded[:, c.observed:])


def forecasting_loss(prediction, target, position_weight=1.0, duration_weight=1.0):
    position = nn.functional.mse_loss(prediction[..., :3], target[..., :3])
    duration = nn.functional.mse_loss(prediction[..., 3], target[..., 3])
    return position_weight * position + duration_weight * duration
