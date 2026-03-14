# STDCA-DA: An integrated deep learning framework for EEG-based depression detection with spatiotemporal dual cross-attention and domain adaptation

This is an official repository related to the paper **"STDCA-DA"**.

- **Paper link:** []  
- **Author:** [Dongxiao Zhang]，[Zhiwei Rao], [Weijun Gu]  
- **Dataset link:** [https://figshare.com/articles/dataset/EEG_Data_New/4244171]

## 1. Project Introductio
We propose STDCA‑DA, an end‑to‑end deep learning framework for EEG‑based depressive disorder (DD) detection. It integrates spatiotemporal dual cross‑attention with adversarial domain adaptation to tackle cross‑subject generalization challenges caused by EEG inter‑subject variability. The model uses parallel convolutional branches to learn disentangled spatiotemporal features, a dual‑attention mechanism for deep spatiotemporal interaction, and adversarial domain adaptation to align subject feature distributions and mitigate domain shifts.
- Core Contributions:
  1. Proposes STDCA‑DA, a novel end‑to‑end deep learning framework integrating spatiotemporal dual cross‑attention with adversarial domain adaptation for robust DD detection.
  2. Aims to address the challenge of limited cross‑subject generalization in EEG‑based depressive disorder (DD) detection, thereby helping to mitigate the adverse effects of inherent inter‑subject variability in EEG signals.
  3. On the public MPHC dataset, our method demonstrates strong performance through ten‑fold cross‑validation, exhibiting promising generalization capability and highlighting its value as a novel approach for EEG‑based DD detection.

## 2. Core Challenges
1. Despite achieving strong performance, the inherent black‑box nature of deep learning still limits the interpretability of the model’s decision‑making process, suggesting that further efforts are needed to enhance transparency and explainability.
2. While cross‑subject generalization has been addressed, broader challenges such as cross‑scenario and cross‑device variability remain unexplored, leaving open questions about the robustness of the framework in diverse real‑world conditions.
3. The evaluation setting follows a transductive unsupervised domain adaptation scenario, where unlabeled target data from the test fold are incorporated during training. Although this improves alignment, it also raises the need for future work to validate inductive settings and ensure broader applicability.

## 3. Parameter Configuration

### 6.1 Environmental Requirements

- **System**: Ubuntu 18.04
- **Software**: Python, PyTorch 1.13.1, CUDA 11.4
- **Hardware**: Intel Xeon(R) Silver 4208 CPU, NVIDIA Tesla V100S, 128GB RAM

### 6.2 Core Training Parameters

| Parameter                | Value                                           |
| ------------------------ | ----------------------------------------------- |
| Learning rate            | 0.0001                                          |
| Batch size               | 32                                              |
| Maximum epochs           | 200                                             |

## 4. Usage Guide

### 4.1 Environment Setup

```bash
# Clone repository
git clone https://github.com/rzw-1/STDCA-DA.git
cd STDCA-DA
```

### 4.2 Data Preparation

1. Dataset structure (example):

  ```plaintext
MPHC_pre_4S/
├── HC/
│   ├── H S1 EO.mat
│   ├── H S2 EO.mat
│   ├── ... 
│   └── H S30 EO.mat
└── MDD/
    ├── MDD S1 EO.mat
    ├── MDD S2 EO.mat
    ├── ...
    └── MDD S34 EO.mat
```
2. The assignment of subjects to each fold’s test set is stored in 10_fold_subjects.txt.


### 4.3 Training and testing Commands

```bash
# Train and test on UCI Dataset
python train_and_test.py
```
