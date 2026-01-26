# Proactive-Nudge-Engine 🚀

Proactive-Nudge-Engine은 구글의 Magic Cue와 같이 사용자의 맥락을 지능적으로 파악하여 필요한 정보를 선제적으로 제안하는 Proactive Nudge(선제적 넛지) 시스템을 구현하는 프로젝트입니다.

## 📌 Project Overview

기존의 어시스턴트가 사용자의 질문에 답하는 방식(Reactive)이었다면, 본 프로젝트는 사용자가 묻기 전에 시스템이 먼저 필요한 조치를 제안(Proactive)하는 것을 목표로 합니다.

### Core Workflow

1. **Triggering & Categorization (MobileBERT)**: 사용자 맥락을 분석하여 넛지가 필요한지 판단함과 동시에, 어떤 종류의 정보(여권, 멤버십코드, 항공 티켓 일정 등)가 필요한지 7가지 카테고리로 분류합니다.
2. **Smart Retrieval (Rule-Based)**: 분류된 카테고리에 맞춰 미리 저장된 사용자 정보(Magic Cue)를 즉각적으로 호출하여 제안합니다.

## 🏗 System Architecture

```text
[ Context Input ] -> [ MobileBERT Classifier ] -> (Category ID) -> [ Smart Retrieval ] -> [ Proactive Nudge UI ]
      (App, SMS)       (Multi-class Trigger)                         (Local DB)            (Card/Bubble)
```

## 💡 Model Selection & Optimization (Google Pixel Style)

본 프로젝트의 아키텍처는 **Google Pixel의 'Android System Intelligence' (Magic Cue)** 와 동일한 설계 철학을 따릅니다. 
생성형 AI(LLM)가 개인정보를 직접 생성할 때 발생할 수 있는 **환각(Hallucination)** 과 **보안(Privacy)** 문제를 원천 차단하기 위해, 실제 상용화된 온디바이스 AI와 같이 **[경량 분류 + 보안 검색]** 구조를 채택했습니다.

### 1. MobileBERT (Triggering)
- **선정 이유**: BERT Base 모델 대비 크기는 1/4, 속도는 5.5배 빠르면서도 유사한 성능을 냅니다.
- **역할**: Pixel의 `System Intelligence` 코어와 같이, 단순한 트리거 유무(Binary)가 아닌 **구체적인 의도(Intent)를 7개 클래스로 정밀 분류**합니다.

### 2. Retrieval System (Nudge Generation)
- **설계 의도**: LLM이 여권 번호를 '상상'해서 생성하는 것은 위험합니다. 본 시스템은 의도가 파악되면 **보안 저장소(Secure Store)** 에서 정확한 실제 데이터를 **인출(Retrieval)** 하는 방식을 사용합니다.
- **장점**: 
  1. **Zero Hallucination**: 100% 정확한 데이터 제공
  2. **Privacy First**: 민감한 개인정보가 모델 가중치에 포함되지 않음
  3. **Ultra-Low Latency**: 생성 과정 없이 즉각적인 UI 표출 가능

## 📂 Project Structure

프로젝트의 효율적인 관리와 학습을 위해 아래와 같은 구조를 유지합니다.

```text
Proactive-Nudge-Engine/
├── data/               # Taskmaster, MultiWOZ 등 데이터셋 저장
├── models/             # Fine-tuned BERT 체크포인트
├── src/                # 핵심 소스 코드
│   ├── preprocess.py   # 데이터 전처리 (카테고리 라벨링 포함)
│   ├── train_bert.py   # MobileBERT 트리거 및 카테고리 분류 학습
│   └── train_gpt2.py   # (Deprecated) 생성형 접근 방식 레거시 코드
├── notebooks/          # 실험 및 시각화 (Jupyter)
├── reports/            # 실험 고찰 리포트
│   ├── 2026-xx-xx_BERT_trigger_test.md
│   └── gpt2_generation_analysis.md
├── assets/             # README나 보고서에 쓸 이미지들
│   ├── bert_loss.png
│   └── architecture.png
├── .gitignore          # 대용량 모델 및 가상환경 제외
├── README.md
└── requirements.txt
```

## 🚀 Setup & Installation

### Prerequisites

- **OS**: macOS
- **GPU**: Apple Silicon (MPS)
- **Package Manager**: Conda (Miniconda or Anaconda)

### Environment Setup

macOS (Apple Silicon) 환경에 최적화된 Python 3.11 환경을 구축합니다.

```bash
# 환경 생성
conda create -n proactive-nudge -c conda-forge python=3.11 -y
conda activate proactive-nudge

# PyTorch for Apple Silicon (MPS)
pip install torch torchvision torchaudio
pip install transformers datasets accelerate evaluate sentencepiece protobuf
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
