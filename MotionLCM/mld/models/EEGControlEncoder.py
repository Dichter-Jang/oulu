import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGControlEncoder(nn.Module):
    """
    EEG -> eeg_emb -> action_latent -> text_emb

    Input:
        x: [B, T, C]
    Output:
        text_emb: [B, 1, D] (D=768)
        action_latent:   [B, d]
        eeg_emb:        [B, H]
    """
    def __init__(self, in_ch=27, hidden=128, latent_dim=32, out_dim=768, normalize=True):
        super().__init__()
        self.normalize = normalize

        self.backbone = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.ReLU(),
        )

        self.to_latent = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

        self.to_text = nn.Linear(latent_dim, out_dim)

    def encode_eeg(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)               # [B,C,T]
        h = self.backbone(x).mean(dim=-1)   # [B,H]
        return h

    def forward(self, x: torch.Tensor):
        eeg_feat = self.encode_eeg(x)                 # [B,H]
        action_latent = self.to_latent(eeg_feat)      # [B,d]
        cond = self.to_text(action_latent)
        if self.normalize:
            cond = F.normalize(cond, dim=-1)
        motion_cond_emb = cond.unsqueeze(1)           # [B,1,D]
        return motion_cond_emb, action_latent, eeg_feat
