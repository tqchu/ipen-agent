import torch, torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps    = eps

    def forward(self, x):
        # x: (B, T, d)
        # 1) compute variance + eps
        var = x.pow(2).mean(-1, keepdim=True).add(self.eps)
        # 2) invert sqrt in one shot
        inv_rms = torch.rsqrt(var)
        # 3) apply scale
        return self.weight * x * inv_rms
