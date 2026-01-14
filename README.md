### Task 1. Processing the EEG signal with foundation models with existing data.

#### data:
1. data\seed_eeg: raw EEG data for 1 subject from SEED dataset
2. data\tuh_eeg: EEG data from The TUH Abnormal EEG Corpus (TUAB): A corpus of EEGs that have been annotated as normal or abnormal.
3. data\synthetic_eeg: EEG data generate by Gaussion Process Joint Model (https://www.pnas.org/doi/10.1073/pnas.1912342117). 3 labels' EEG date.

#### notebooks:
1. notebooks\01_eegformer_seed: using EEGFormer processing SEED dataset (without training)
2. notebooks\02_brainbert_seed: using BrainBERT processing SEED dataset (use the pretrained weight)
3. notebooks\04_eegformer_synthetic: using EEGFormer processing synthetic dataset (with training, results in outputs folder)
4. notebooks\05_brainbert_synthetic: using BrainBERT processing synthetic dataset (use the pretrained weight)
5. notebooks\06_eegformer_tuh:using EEGFormer processing TUH dataset (with training, results in outputs folder)
6. notebooks\07_brainbert_tuh:using BrainBERT processing TUH dataset (use the pretrained weight)

#### models:
1. BrainBERT folder contains the code from: https://github.com/czlwang/BrainBERT
2. EEGformer folder contains the code from: https://github.com/FENRlR/EEGformer

### Task 2: Using MotionLCM to generate some controllable motions(in Colab notebook), and try to link the EEG with MotionLCM.

#### model:
MotionLCM folder contains the code from: https://github.com/Dai-Wenxun/MotionLCM

#### notebooks:
notebooks\03_motionlcm_eeg:
1. notebooks\03_motionlcm_eeg\MotionLCM_demo.ipynb: Reproduction of the official MotionLCM demo script, demonstrating controllable human motion generation and verifying the baseline functionality of the model.
2. notebooks\03_motionlcm_eeg\MotionLCM_EEG.ipynb: An experimental notebook exploring the integration of EEG signals with MotionLCM, aiming to investigate the feasibility of conditioning motion generation on neural signals.

### Note:
1. Due to the large size of the datasets and pretrained model weights, they are not included in this repository.
2. Detailed implementation logic, experimental design, and intermediate results are documented within the notebooks.