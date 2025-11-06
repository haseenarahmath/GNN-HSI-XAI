import math
import torch
import torch.nn as nn

# Minimal PairNorm fallback
class PairNorm(nn.Module):
    def __init__(self, mode: str = "None", scale: float = 1.0):
        super().__init__()
        self.mode = mode
        self.scale = scale
    def forward(self, x):
        if self.mode == "None":
            return x
        x_c = x - x.mean(dim=0, keepdim=True)
        if self.mode == "PN":
            col_norm = x_c.pow(2).sum(dim=1, keepdim=True).sqrt().mean()
            return self.scale * x_c / (col_norm + 1e-6)
        if self.mode == "PN-SI":
            row_norm = x_c.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-6
            return self.scale * x_c / row_norm
        return x

class GraphConv(nn.Module):
    """Dense-adj graph conv: H = A * (X W). A expected pre-normalized (e.g., row-normalized)."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in = self.weight.size(0)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, x, adj):
        h = x @ self.weight
        out = torch.matmul(adj, h)
        return out + self.bias if self.bias is not None else out

class TinyDeepGCN(nn.Module):
    """Deep GCN stack with PairNorm and dropout; forward(x, adj) -> logits."""
    def __init__(self, in_dim, hid, out_dim, n_layers=3, dropout=0.1, norm_mode="PN-SI", norm_scale=1.0):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hid)
        self.convs = nn.ModuleList([GraphConv(hid, hid) for _ in range(n_layers)])
        self.norm = PairNorm(norm_mode, norm_scale)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(hid, out_dim)

    def forward(self, x, adj):
        h = self.fc_in(x)
        for conv in self.convs:
            h = self.drop(h)
            h = conv(h, adj)
            h = self.norm(h)
            h = self.relu(h)
        return self.fc_out(h)
