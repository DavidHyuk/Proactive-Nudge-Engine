# Proactive-Nudge-Engine 🚀

Proactive-Nudge-Engine은 구글의 Magic Cue와 같이 사용자의 맥락을 지능적으로 파악하여 필요한 정보를 선제적으로 제안하는 Proactive Nudge(선제적 넛지) 시스템을 구현하는 프로젝트입니다.

## 📌 Project Overview

기존의 어시스턴트가 사용자의 질문에 답하는 방식(Reactive)이었다면, 본 프로젝트는 사용자가 묻기 전에 시스템이 먼저 필요한 조치를 제안(Proactive)하는 것을 목표로 합니다.

### Core Workflow

1. **Triggering (BERT)**: 현재 대화나 상황 맥락을 분석하여 "지금 넛지가 필요한 타이밍인가?"를 분류합니다.
2. **Generation (mT5)**: 넛지가 필요하다고 판단되면, 다국어 지원이 가능한 mT5 모델을 통해 상황에 맞는 최적의 제안 문구를 생성합니다.

## 🏗 System Architecture

```text
[ Context Input ] -> [ BERT Classifier ] -> (Nudge Needed?) -> [ mT5 Generator ] -> [ Proactive Cue ]
      (App, SMS,           (Trigger)                               (Content)             (UI Card)
       Calendar)
```

## 📂 Project Structure

DGX 서버에서의 효율적인 관리와 학습을 위해 아래와 같은 구조를 유지합니다.

```text
Proactive-Nudge-Engine/
├── data/               # Taskmaster, MultiWOZ 등 데이터셋 저장
├── models/             # Fine-tuned BERT & mT5 체크포인트
├── src/                # 핵심 소스 코드
│   ├── preprocess.py   # 데이터 전처리 스크립트
│   ├── train_bert.py   # 트리거 분류 모델 학습
│   └── train_mt5.py    # 넛지 생성 모델 학습
├── notebooks/          # 실험 및 시각화 (Jupyter)
├── reports/            # 실험 고찰 리포트
│   ├── 2026-xx-xx_BERT_trigger_test.md
│   └── mT5_generation_analysis.md
├── assets/             # README나 보고서에 쓸 이미지들
│   ├── bert_loss.png
│   └── architecture.png
├── scripts/            # DGX 서버 실행용 쉘 스크립트
├── .gitignore          # 대용량 모델 및 가상환경 제외
├── README.md
└── requirements.txt
```

## 🚀 Setup & Installation

### Prerequisites

- **OS**: Linux (aarch64 recommended for DGX)
- **GPU**: NVIDIA GPU + CUDA 13.0 (cu130)
- **Package Manager**: Conda (Miniconda or Anaconda)

### Environment Setup

DGX 서버(ARM 아키텍처) 환경에 최적화된 Python 3.11 환경을 구축합니다.

```bash
# 환경 생성
conda create -n proactive-nudge -c conda-forge python=3.11 -y
conda activate proactive-nudge

# Libs (CUDA 13.0 대응 최신 버전)
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install transformers==4.57.6 datasets==4.5.0 accelerate==1.12.0 evaluate==0.4.6 sentencepiece==0.2.1 protobuf
```

## 📊 Datasets

이 프로젝트는 아래의 데이터셋을 활용하여 모델을 학습합니다.

| Dataset | Primary Use | Source |
| :--- | :--- | :--- |
| Taskmaster | Intent & Next action prediction | Google Research |
| MultiWOZ | Multi-domain task dialogue | Hugging Face |
| SMCalFlow | Calendar-based event flows | Microsoft |

## 📜 License

This project is licensed under the MIT License.
