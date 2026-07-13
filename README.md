# High-Efficiency Spatial-Temporal Hybrid Architecture for 1D ECG Anomaly Detection: A Theoretical Framework for Edge Environments
```markdown


This repository contains the official, verified open-source Python implementation and hardware profiling pipeline for our proposed hybridized **Conv1D-GRU** framework. The architecture is engineered explicitly for low-power edge microcontrollers, achieving significant reductions in computational complexity and floating-point operations (FLOPs) while maintaining robust classification accuracy on 1D time-series electrocardiogram (ECG) data.

---

##  Project Overview

Resource-constrained edge hardware demands optimized deep learning models that minimize parameter count and memory latency without degrading diagnostic accuracy. This project provides a complete end-to-end evaluation pipeline that contrasts classic baselines against our proposed spatial-temporal framework across various downsampling pooling factors ($P \in \{4, 8, 32, 128\}$).

### Evaluated Architectures:
1. **Baseline_CNN**: Dual-layer 1D Convolution with Global Average Pooling.
2. **Baseline_LSTM**: Standard Long Short-Term Memory recurrent network.
3. **Baseline_GRU**: Gated Recurrent Unit network for temporal dependency extraction.
4. **Hybrid_Conv1D_GRU (Proposed)**: An optimized pipeline leveraging a light 1D spatial feature extraction layer coupled with parameterized downsampling layers feeding directly into a tight recurrent temporal gate.

---

##  Repository Structure

```text
├── data/
│   └── mitbih_train.csv      # Clinical dataset (User must download manually)
├── main.py                   # Unified training, evaluation, and profiling script
└── README.md                 # Project documentation and reproduction guide
```

---

## 📊 Dataset Setup & Preprocessing

The pipeline utilizes the widely adopted, preprocessed 125 Hz variant of the MIT-BIH Arrhythmia Database originally curated by Kachuee et al. Each heartbeat segment is zero-padded to match exactly 187 architectural time steps.

### Download Instructions:
1. Navigate to the Kaggle data repository: [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).
2. Download the archive and extract `mitbih_train.csv`.
3. Create a folder named `data` in your local project root directory.
4. Place the `mitbih_train.csv` file inside the `data/` directory.

### Automated Processing Pipeline:
* **Binary Mapping**: Labels greater than 0 are programmatically mapped to `1` (Anomaly) to establish an objective edge validation environment; normal records are mapped to `0`.
* **Data Split**: Implements a strict, reproducible train-test split (80% train, 20% test) stratified by target distribution to guarantee standard, non-overlapping evaluation metrics.

---

## 🛠️ Installation & Prerequisites

To execute the profiling code locally, ensure your machine has an initialized Python 3.8+ ecosystem. Install the verified versions of the required software dependencies via your terminal:

```bash
pip install tensorflow numpy pandas scikit-learn
```

---

## 🚀 Execution & Reproducibility Pipeline

The script handles compilation, callbacks, automated convergence checks, and inference hardware latency calculation sequentially. Execute the core run command via your terminal console:

```bash
python main.py
```

### Metrics Tracked Automatically:
* **Validation Convergence**: Outfitted with a rigorous `EarlyStopping` callback (patience threshold = 10 epochs) tracking validation loss deviations to prevent overfitting.
* **Inference Latency Execution**: Simulates real-time edge processing loops by executing high-sample (N = 1000) batch evaluation blocks to calculate clean, normalized per-sample latency metrics in milliseconds (ms).
* **Classification Accuracy & F1-Score**: Calculates out-of-sample metrics to guarantee high performance against severe clinical class imbalances.

---

##  Citation & Scientific References

If you build upon this architecture or utilize the provided benchmarks for academic research, please ensure proper attribution to the baseline preprocessed framework:

```text
@inproceedings{kachuee2018ecg,
  title={ECG heartbeat classification: A deep transferable representation},
  author={Kachuee, Mohammad and Fazeli, Shayan and Sarrafzadeh, Majid},
  booktitle={2018 IEEE Nuclear Science Symposium and Medical Imaging Conference (NSS/MIC)},
  pages={1--4},
  year={2018},
  organization={IEEE}
}
```

```
