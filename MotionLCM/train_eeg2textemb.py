import os
import numpy as np
import torch
import torch.nn.functional as F

from omegaconf import OmegaConf
from mld.config import parse_args
from mld.data.get_data import get_dataset
from mld.models.modeltype.mld import MLD
from mld.utils.utils import set_seed
from mld.models.eeg_encoder import EEGEncoder

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

    # Load MotionLCM model (teacher text encoder inside)
    dataset = get_dataset(cfg)
    model = MLD(cfg, dataset).to(device).eval()
    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.requires_grad_(False)

    # Build EEG encoder (student)
    eeg_encoder = EEGEncoder(in_ch=27, hidden=128, out_dim=768).to(device).train()
    opt = torch.optim.AdamW(eeg_encoder.parameters(), lr=1e-3, weight_decay=1e-4)

    # Cache teacher embeddings (only 3 labels)
    teacher = {}
    with torch.no_grad():
        for label, prompt in label_to_prompt.items():
            emb = model.text_encoder([prompt]).to(device)  # [1,1,768]
            teacher[label] = emb.detach()

    # Train loop (Toy)
    steps = 2000
    for step in range(1, steps + 1):
        label = list(label_to_prompt.keys())[(step - 1) % 3]

        eeg = np.load(label_to_eeg[label]).astype(np.float32)  # [151,27]
        x = torch.from_numpy(eeg).unsqueeze(0).to(device)      # [1,151,27]

        pred = eeg_encoder(x)          # [1,1,768]
        tgt = teacher[label]           # [1,1,768]

        loss = F.mse_loss(pred, tgt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"step {step}: label={label} loss={loss.item():.6f}")

    os.makedirs("experiments_eeg", exist_ok=True)
    torch.save({"eeg_encoder": eeg_encoder.state_dict()}, "experiments_eeg/eeg2textemb.ckpt")
    print("Saved to experiments_eeg/eeg2textemb.ckpt")

if __name__ == "__main__":
    main()
