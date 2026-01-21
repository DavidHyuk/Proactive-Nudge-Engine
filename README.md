# Proactive-Nudge-Engine 🚀

Proactive-Nudge-Engine is a project that implements a Proactive Nudge system, which intelligently analyzes user context and proactively suggests necessary information, similar to Google's Magic Cue.

## 📌 Project Overview

While traditional assistants respond to user queries (Reactive), this project aims to have the system proactively suggest necessary actions before the user asks (Proactive).

### Core Workflow

1. **Triggering (BERT)**: Analyzes the current conversation or situational context to classify whether "now is the timing for a nudge?"

2. **Generation (mT5)**: If a nudge is deemed necessary, generates optimal suggestion text tailored to the situation using the multilingual-supporting mT5 model.

## 🏗 System Architecture

```text
[ Context Input ] -> [ BERT Classifier ] -> (Nudge Needed?) -> [ mT5 Generator ] -> [ Proactive Cue ]
      (App, SMS,           (Trigger)                               (Content)             (UI Card)
       Calendar)
```

## 📂 Project Structure

To maintain efficient management and training on DGX servers, the following structure is maintained.

```text
Proactive-Nudge-Engine/
├── data/               # Dataset storage such as Taskmaster, MultiWOZ
├── models/             # Fine-tuned BERT & mT5 checkpoints
├── src/                # Core source code
│   ├── preprocess.py   # Data preprocessing script
│   ├── train_bert.py   # Training script for trigger classification model
│   └── train_mt5.py    # Training script for nudge generation model
├── notebooks/          # Experiments and visualization (Jupyter)
├── reports/            # Experiment reports
│   ├── 2026-xx-xx_BERT_trigger_test.md
│   └── mT5_generation_analysis.md
├── assets/             # Images for README or reports
│   ├── bert_loss.png
│   └── architecture.png
├── scripts/            # Shell scripts for DGX server execution
├── .gitignore          # Exclude large models and virtual environments
├── README.md
└── requirements.txt
```

## 🚀 Setup & Installation

### Prerequisites

- **OS**: Linux (aarch64 recommended for DGX)
- **GPU**: NVIDIA GPU + CUDA 13.0 (cu130)
- **Package Manager**: Conda (Miniconda or Anaconda)

### Environment Setup

Build a Python 3.11 environment optimized for DGX server (ARM architecture).

```bash
# Create environment
conda create -n proactive-nudge -c conda-forge python=3.11 -y
conda activate proactive-nudge

# Libs (Latest versions compatible with CUDA 13.0)
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install transformers==4.57.6 datasets==4.5.0 accelerate==1.12.0 evaluate==0.4.6 sentencepiece==0.2.1 protobuf
```

## 📊 Datasets

This project uses the following datasets to train the models.

| Dataset | Primary Use | Source |
| :--- | :--- | :--- |
| Taskmaster | Intent & Next action prediction | Google Research |
| MultiWOZ | Multi-domain task dialogue | Hugging Face |
| SMCalFlow | Calendar-based event flows | Microsoft |

## 📜 License

This project is licensed under the MIT License.
