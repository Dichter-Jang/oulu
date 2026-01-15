import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf
from mld.config import parse_args
from mld.data.get_data import get_dataset
from mld.models.modeltype.mld import MLD
from mld.utils.utils import set_seed
from mld.models.EEGControlEncoder import EEGControlEncoder

label_to_prompt = {
    "label1": "a person walks forward steadily",
    "label2": "a person squats down and stands up",
    "label3": "a person turns around in place",
}
label_to_eeg = {
    "label1": "data_eeg/neural_data_label1_1.npy",
    "label2": "data_eeg/neural_data_label2_1.npy",
    "label3": "data_eeg/neural_data_label3_1.npy",
}

def main():
    cfg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.SEED_VALUE)

    if cfg.model.get("noise_optimizer") is not None:
        cfg.model.noise_optimizer = None

    # Load teacher MotionLCM model
    dataset = get_dataset(cfg)
    model = MLD(cfg, dataset).to(device).eval()
    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.requires_grad_(False)

    # Build student EEG control encoder
    latent_dim = 32
    eeg_control = EEGControlEncoder(
        in_ch=27, hidden=128, latent_dim=latent_dim, out_dim=768, normalize=True
    ).to(device).train()

    opt = torch.optim.AdamW(eeg_control.parameters(), lr=1e-3, weight_decay=1e-4)

    teacher = {}
    with torch.no_grad():
        for label, prompt in label_to_prompt.items():
            emb = model.text_encoder([prompt]).to(device)  # [1,1,768]
            # if student normalizes, normalize teacher too
            emb = F.normalize(emb, dim=-1)
            teacher[label] = emb.detach()

    # Train loop (Toy)
    steps = 2000
    keys = list(label_to_prompt.keys())

    for step in range(1, steps + 1):
        label = keys[(step - 1) % len(keys)]

        eeg = np.load(label_to_eeg[label]).astype(np.float32)     # [151,27]
        x = torch.from_numpy(eeg).unsqueeze(0).to(device)         # [1,151,27]

        pred_cond, action_latent, eeg_feat = eeg_control(x)       # [1,1,768], [1,d], [1,hidden]
        tgt = teacher[label]                                      # [1,1,768]

        loss_align = F.mse_loss(pred_cond, tgt)
        loss_latent = (action_latent ** 2).mean()
        loss = loss_align + 1e-4 * loss_latent

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            with torch.no_grad():
                cos = F.cosine_similarity(pred_cond.flatten(1), tgt.flatten(1)).mean().item()
            print(
                f"step {step}: label={label} "
                f"loss={loss.item():.6f} align={loss_align.item():.6f} "
                f"cos={cos:.4f} | eeg_feat={tuple(eeg_feat.shape)} latent={tuple(action_latent.shape)}"
            )

    # Save checkpoint
    os.makedirs("experiments_eeg", exist_ok=True)
    ckpt_path = "experiments_eeg/eeg_control_encoder_v2.ckpt"
    torch.save(
        {
            "eeg_control_encoder": eeg_control.state_dict(),
            "latent_dim": latent_dim,
            "hidden": 128,
            "in_ch": 27,
            "out_dim": 768,
            "normalize": True,
            "label_to_prompt": label_to_prompt,
        },
        ckpt_path,
    )
    print(f"Saved to {ckpt_path}")


if __name__ == "__main__":
    main()
