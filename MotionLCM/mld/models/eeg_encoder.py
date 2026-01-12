import torch
import torch.nn as nn

class EEGEncoder(nn.Module):
    """
    Input:  x [B, T, C]  (e.g. [1, 151, 27])
    Output: y [B, 1, 768] to match text_emb
    """
    def __init__(self, in_ch=27, hidden=128, out_dim=768):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        h = self.backbone(x)     # [B, hidden, T]
        h = h.mean(dim=-1)       # [B, hidden]
        y = self.proj(h)         # [B, 768]
        return y.unsqueeze(1)    # [B, 1, 768]
