# Continuous Blood Pressure Estimation from ECG and PPG Signals via a Novel Multi-Scale Cross-Attention Architecture with Cross-Scale Skip Connections

This is an official repository related to the paper **"CBPE-Net"**.

- **Paper link:** []  
- **Author:** [Dongxiao Zhang]，[Shunbin Chen], [Jialiang Xie]  
- **Dataset link:** [https://www.kaggle.com/datasets/mkachuee/BloodPressureDataset]

## 1. Project Introductio
We propose CBPE-Net, a novel deep learning framework for continuous blood pressure estimation. The model integrates cross-scale skip connections (CSSC) to reinforce the fusion of shallow details with deep semantic features, and employs a multi-scale cross-attention mechanism (MSCA) to dynamically guide interactions across different feature scales. Combined with a sliding-window strategy (step size 0.008 s), the model outputs the corresponding instantaneous BP value at each time step, realizing true continuous prediction
- Core Contributions:
  1. Moving beyond the limitations of discrete BP prediction, our model provides a real-time, continuous BP values. This advancement is critically important for clinical practice, as it facilitates detailed hemodynamic assessment and supports more informed, rapid therapeutic interventions.
  2. The proposed PIP framework eliminates the dependence of existing methods on full-cycle signal acquisition, thereby enabling real-time, continuous BP prediction.
  3. With the support of the PIP, the proposed  CBPE-Net enables accurate and real-time continuous BP monitoring. Experimental results demonstrate that the model's performance meets the AAMI standard and achieves a Grade C rating under the stringent BHS protocol. Furthermore, in comparative assessments, our model consistently outperforms established classical machine learning and contemporary deep learning approaches on all evaluated metrics.

## 2. Core Challenges
1. Existing cuffless BP estimation methods typically produce only beat-level outputs, limiting their ability to support real-time, continuous monitoring—an essential requirement for dynamic hemodynamic assessment in clinical settings. These limitations underscore the urgent need for an end-to-end model capable of high–temporal-resolution continuous BP prediction.
2. ECG and PPG signals contain rich information across multiple temporal scales (e.g., heart rate, pulse transit time, waveform morphology). Capturing and integrating these cross-scale features effectively is technically challenging.
3. Deep learning models often operate as "black boxes," making it unclear how ECG signals specifically contribute to blood pressure (BP) estimation. Understanding the underlying physiological mechanisms remains an open challenge.

## 3. Parameter Configuration

### 6.1 Environmental Requirements

- **System**: Ubuntu 18.04
- **Software**: Python 3.8, PyTorch 2.0.1, CUDA 11.4
- **Hardware**: Intel(R) Xeon(R) Silver 4208 CPU, NVIDIA Tesla V100S (24GB VRAM), 126GB memory

### 6.2 Core Training Parameters

| Parameter                | Value                                           |
| ------------------------ | ----------------------------------------------- |
| Learning rate            | 0.1                                             |
| Batch size               | 300                                             |
| Epochs                   | 10                                              |
| Accumulation steps       | 10                                              |

## 4. Usage Guide

### 4.1 Environment Setup

```bash
# Clone repository
git clone https://github.com/abin7777/CBPE-Net.git
cd CBPE-Net

# Install dependencies
pip install -r requirements.txt
```

### 4.2 Data Preparation

1. Dataset structure (example):


   ```plaintext
   UCI_Dataset/
     ├── part1.mat      
     ├── ...  
     └── part12.mat  
   ```

### 4.3 Training and testing Commands

```bash
# Train and test on UCI Dataset
python train_and_test.py
```

### 4.4 Visualization

```bash
# Visualize results 
python visualize.py
```
