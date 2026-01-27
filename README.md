# Proactive-Nudge-Engine 🚀

Proactive-Nudge-Engine is a project that implements a Proactive Nudge system, similar to Google's Magic Cue, which intelligently identifies the user's context and proactively suggests necessary information.

## 📌 Project Overview

While traditional assistants respond to user questions (Reactive), this project aims to have the system proactively suggest necessary actions before the user asks (Proactive).

### Core Workflow

1. **Triggering & Categorization (MobileBERT)**: Analyzes user context to determine if a nudge is needed and simultaneously classifies what type of information (passport, membership code, flight schedule, etc.) is required into 7 categories.
2. **Smart Retrieval (Rule-Based)**: Instantly retrieves and suggests pre-stored user information (Magic Cue) based on the classified category.

## 🏗 System Architecture

```text
[ Context Input ] -> [ MobileBERT Classifier ] -> (Category ID) -> [ Smart Retrieval ] -> [ Proactive Nudge UI ]
      (App, SMS)       (Multi-class Trigger)                         (Local DB)            (Card/Bubble)
```

## 💡 Model Selection & Optimization (Google Pixel Style)

The architecture of this project follows the same design philosophy as **Google Pixel's 'Android System Intelligence' (Magic Cue)**. To fundamentally prevent **Hallucination** and **Privacy** issues that can occur when Generative AI (LLM) directly generates personal information, we adopted a **[Lightweight Classification + Secure Retrieval]** structure, similar to actual commercial on-device AI.

### 1. MobileBERT (Triggering)
- **Reason for Selection**: It is 1/4 the size and 5.5x faster than the BERT Base model while achieving similar performance.
- **Role**: Like the `System Intelligence` core of Pixel, it precisely classifies **specific intents into 7 classes**, rather than just binary trigger presence.

### 2. Retrieval System (Nudge Generation)
- **Design Intent**: It is dangerous for an LLM to 'imagine' and generate a passport number. This system uses a method of **Retrieving** accurate actual data from a **Secure Store** once the intent is identified.
- **Advantages**: 
  1. **Zero Hallucination**: Provides 100% accurate data
  2. **Privacy First**: Sensitive personal information is not included in model weights
  3. **Ultra-Low Latency**: Immediate UI display possible without generation process

## 📂 Project Structure

The following structure is maintained for efficient project management and training.

```text
Proactive-Nudge-Engine/
├── data/               # Storage for Taskmaster, MultiWOZ, etc. datasets
├── models/             # Fine-tuned BERT checkpoints
├── src/                # Core source code
│   ├── preprocess.py   # Data preprocessing (includes category labeling)
│   ├── train_bert.py   # MobileBERT trigger and category classification training
│   ├── inference.py    # On-device inference engine simulation with smart retrieval
│   └── export_onnx.py  # Script to export trained models to ONNX for Android
├── android_app/        # Android demo application (Jetpack Compose)
├── notebooks/          # Experiments and visualization (Jupyter)
├── reports/            # Experiment review reports
│   ├── 2026-xx-xx_BERT_trigger_test.md
│   └── gpt2_generation_analysis.md
├── assets/             # Images for README or reports
│   ├── bert_loss.png
│   └── architecture.png
├── .gitignore          # Exclude large models and virtual environments
├── README.md
└── requirements.txt
```

## 🚀 Setup & Installation

### Prerequisites

- **OS**: macOS
- **GPU**: Apple Silicon (MPS)
- **Package Manager**: Conda (Miniconda or Anaconda)

### Environment Setup

Set up a Python 3.11 environment optimized for macOS (Apple Silicon).

```bash
# Create environment
conda create -n proactive-nudge -c conda-forge python=3.11 -y
conda activate proactive-nudge

# PyTorch for Apple Silicon (MPS)
pip install torch torchvision torchaudio
pip install transformers datasets accelerate evaluate sentencepiece protobuf
```

## 📊 Datasets

This project uses the following datasets to train the model.

| Dataset | Primary Use | Source |
| :--- | :--- | :--- |
| Taskmaster | Intent & Next action prediction | Google Research |
| MultiWOZ | Multi-domain task dialogue | Hugging Face |
| SMCalFlow | Calendar-based event flows | Microsoft |

## 📜 License

This project is licensed under the MIT License.
