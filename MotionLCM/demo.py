import os
import pickle
import sys
import datetime
import logging
import os.path as osp

from omegaconf import OmegaConf

import torch
import numpy as np

from mld.config import parse_args
from mld.data.get_data import get_dataset
from mld.models.modeltype.mld import MLD
from mld.models.modeltype.vae import VAE
from mld.utils.utils import set_seed, move_batch_to_device
from mld.data.humanml.utils.plot_script import plot_3d_motion
from mld.utils.temos_utils import remove_padding
from mld.models.eeg_encoder import EEGEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_example_hint_input(text_path: str) -> tuple:
    with open(text_path, "r") as f:
        lines = f.readlines()

    n_frames, control_type_ids, control_hint_ids = [], [], []
    for line in lines:
        s = line.strip()
        n_frame, control_type_id, control_hint_id = s.split(' ')
        n_frames.append(int(n_frame))
        control_type_ids.append(int(control_type_id))
        control_hint_ids.append(int(control_hint_id))

    return n_frames, control_type_ids, control_hint_ids


def load_example_input(text_path: str) -> tuple:
    with open(text_path, "r") as f:
        lines = f.readlines()

    texts, lens = [], []
    for line in lines:
        s = line.strip()
        s_l = s.split(" ")[0]
        s_t = s[(len(s_l) + 1):]
        lens.append(int(s_l))
        texts.append(s_t)
    return texts, lens


def main():
    cfg = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(cfg.SEED_VALUE)

    name_time_str = osp.join(cfg.NAME, "demo_" + datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    cfg.output_dir = osp.join(cfg.TEST_FOLDER, name_time_str)
    vis_dir = osp.join(cfg.output_dir, 'samples')
    os.makedirs(cfg.output_dir, exist_ok=False)
    os.makedirs(vis_dir, exist_ok=False)

    steam_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(osp.join(cfg.output_dir, 'output.log'))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        handlers=[steam_handler, file_handler])
    logger = logging.getLogger(__name__)

    OmegaConf.save(cfg, osp.join(cfg.output_dir, 'config.yaml'))

    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))

    # Step 1: Check if the checkpoint is VAE-based.
    is_vae = False
    vae_key = 'vae.skel_embedding.weight'
    if vae_key in state_dict:
        is_vae = True
    logger.info(f'Is VAE: {is_vae}')

    # Step 2: Check if the checkpoint is MLD-based.
    is_mld = False
    mld_key = 'denoiser.time_embedding.linear_1.weight'
    if mld_key in state_dict:
        is_mld = True
    logger.info(f'Is MLD: {is_mld}')

    # Step 3: Check if the checkpoint is LCM-based.
    is_lcm = False
    lcm_key = 'denoiser.time_embedding.cond_proj.weight'  # unique key for CFG
    if lcm_key in state_dict:
        is_lcm = True
        time_cond_proj_dim = state_dict[lcm_key].shape[1]
        cfg.model.denoiser.params.time_cond_proj_dim = time_cond_proj_dim
    logger.info(f'Is LCM: {is_lcm}')

    # Step 4: Check if the checkpoint is Controlnet-based.
    cn_key = "controlnet.controlnet_cond_embedding.0.weight"
    is_controlnet = True if cn_key in state_dict else False
    cfg.model.is_controlnet = is_controlnet
    logger.info(f'Is Controlnet: {is_controlnet}')

    if is_mld or is_lcm or is_controlnet:
        target_model_class = MLD
    else:
        target_model_class = VAE

    if cfg.optimize:
        assert cfg.model.get('noise_optimizer') is not None
        cfg.model.noise_optimizer.params.optimize = True
        logger.info('Optimization enabled. Set the batch size to 1.')
        logger.info(f'Original batch size: {cfg.TEST.BATCH_SIZE}')
        cfg.TEST.BATCH_SIZE = 1

    dataset = get_dataset(cfg)
    model = target_model_class(cfg, dataset)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    logger.info(model.load_state_dict(state_dict))

    # ---------------- EEG encoder ----------------
    eeg_ckpt_path = "experiments_eeg/eeg2textemb.ckpt"
    use_eeg = osp.exists(eeg_ckpt_path)
    eeg_encoder = None
    if use_eeg:
        eeg_encoder = EEGEncoder(in_ch=27, hidden=128, out_dim=768).to(device).eval()
        eeg_ckpt = torch.load(eeg_ckpt_path, map_location="cpu")
        eeg_encoder.load_state_dict(eeg_ckpt["eeg_encoder"])
        eeg_encoder.requires_grad_(False)
        logger.info(f"Loaded EEG encoder from {eeg_ckpt_path}")
    else:
        logger.info(f"EEG encoder ckpt not found at {eeg_ckpt_path}, fallback to text-only")
    # --------------------------------------------------------

    FPS = eval(f"cfg.DATASET.{cfg.DATASET.NAME.upper()}.FRAME_RATE")

    if cfg.example is not None and not is_controlnet:
        text, length = load_example_input(cfg.example)
        for t, l in zip(text, length):
            logger.info(f"{l}: {t}")

        def pick_eeg_path(prompt: str) -> str:
          p = prompt.lower()
          if "walk" in p:
              return "data_eeg/neural_data_label1_1.npy"
          if "squat" in p:
              return "data_eeg/neural_data_label2_1.npy"
          if "turn" in p:
              return "data_eeg/neural_data_label3_1.npy"
          # default
          return "data_eeg/neural_data_label1_1.npy"

        batch = {"length": length, "text": text}

        # If EEG encoder is available, build eeg_emb for each sample in this batch
        if eeg_encoder is not None:
            forced_paths = [
                "data_eeg/neural_data_label1_1.npy",
                "data_eeg/neural_data_label2_1.npy",
                "data_eeg/neural_data_label3_1.npy",
            ]
            eeg_list = []
            eeg_paths = []
            # for t in text:
            for i, t in enumerate(text):
                eeg_path = pick_eeg_path(t)
                # eeg_path = forced_paths[i % len(forced_paths)]
                eeg_paths.append(eeg_path)
                eeg = np.load(eeg_path).astype(np.float32)          # [151,27]
                eeg_list.append(torch.from_numpy(eeg))              # [151,27]

            eeg_batch = torch.stack(eeg_list, dim=0).to(device)     # [B,151,27]
            with torch.no_grad():
                eeg_emb = eeg_encoder(eeg_batch)                    # [B,1,768]

            batch["eeg_emb"] = eeg_emb
            logger.info(f"Using EEG conditioning. Paths: {eeg_paths}")
            logger.info(f"EEG emb shape: {tuple(eeg_emb.shape)}")


        for rep_i in range(cfg.replication):
            with torch.no_grad():
                joints = model(batch)[0]

            num_samples = len(joints)
            for i in range(num_samples):
                res = dict()
                pkl_path = osp.join(vis_dir, f"sample_id_{i}_length_{length[i]}_rep_{rep_i}.pkl")
                res['joints'] = joints[i].detach().cpu().numpy()
                res['text'] = text[i]
                res['length'] = length[i]
                res['hint'] = None
                with open(pkl_path, 'wb') as f:
                    pickle.dump(res, f)
                logger.info(f"Motions are generated here:\n{pkl_path}")

                if not cfg.no_plot:
                    plot_3d_motion(pkl_path.replace('.pkl', '.mp4'), joints[i].detach().cpu().numpy(), text[i], fps=FPS)

    else:
        test_dataloader = dataset.test_dataloader()
        for rep_i in range(cfg.replication):
            for batch_id, batch in enumerate(test_dataloader):
                batch = move_batch_to_device(batch, device)
                with torch.no_grad():
                    joints, joints_ref = model(batch)

                num_samples = len(joints)
                text = batch['text']
                length = batch['length']
                if 'hint' in batch:
                    hint, hint_mask = batch['hint'], batch['hint_mask']
                    hint = dataset.denorm_spatial(hint) * hint_mask
                    hint = remove_padding(hint, lengths=length)
                else:
                    hint = None

                for i in range(num_samples):
                    res = dict()
                    pkl_path = osp.join(vis_dir, f"batch_id_{batch_id}_sample_id_{i}_length_{length[i]}_rep_{rep_i}.pkl")
                    res['joints'] = joints[i].detach().cpu().numpy()
                    res['text'] = text[i]
                    res['length'] = length[i]
                    res['hint'] = hint[i].detach().cpu().numpy() if hint is not None else None
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(res, f)
                    logger.info(f"Motions are generated here:\n{pkl_path}")

                    if not cfg.no_plot:
                        plot_3d_motion(pkl_path.replace('.pkl', '.mp4'), joints[i].detach().cpu().numpy(),
                                       text[i], fps=FPS, hint=hint[i].detach().cpu().numpy() if hint is not None else None)

                    if rep_i == 0:
                        res['joints'] = joints_ref[i].detach().cpu().numpy()
                        with open(pkl_path.replace('.pkl', '_ref.pkl'), 'wb') as f:
                            pickle.dump(res, f)
                        logger.info(f"Motions are generated here:\n{pkl_path.replace('.pkl', '_ref.pkl')}")
                        if not cfg.no_plot:
                            plot_3d_motion(pkl_path.replace('.pkl', '_ref.mp4'), joints_ref[i].detach().cpu().numpy(),
                                           text[i], fps=FPS, hint=hint[i].detach().cpu().numpy() if hint is not None else None)


if __name__ == "__main__":
    main()
