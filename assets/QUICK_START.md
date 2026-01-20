# Proactive Nudge Engine - Quick Start Guide 🚀

본 문서는 사용된 데이터셋과 모델 학습/평가 방법에 대해 설명합니다.

---

## 1. 데이터셋 (Datasets)

Proactive Nudge Engine 학습을 위해 여러 대화형 데이터셋을 지원합니다. 시스템은 다음 우선순위에 따라 사용 가능한 데이터셋을 자동으로 감지하고 로드합니다:

1.  **DailyDialog** (기본/권장)
    - 고품질의 일상 대화 데이터입니다.
2.  **DialogSum** (대체)
    - 실생활 대화 요약 데이터셋입니다.
3.  **SAMsum** (대체)
    - 메신저 스타일의 대화 데이터입니다.

만약 실제 데이터셋이 발견되지 않으면, 시스템은 자동으로 **합성 데이터**(1,000개 이상)를 생성하여 사용합니다.

**다운로드 명령어:**
```bash
python src/download_data.py
```

---

## 2. 학습 방법 (Training)

Trigger(BERT)와 Generator(mT5) 모델을 각각 학습할 수 있습니다. `--exp_name` 인자를 사용하여 실험을 관리하세요.

### A. Trigger 모델 (BERT)
**"지금 넛지가 필요한가?"** 를 판단하는 이진 분류기를 학습합니다.

```bash
python src/train_bert.py --exp_name my_experiment_v1
```
- **입력**: 사용자 맥락 (Context)
- **출력**: 1 (넛지 필요) / 0 (넛지 불필요)
- **데이터 분할**: 90% 학습 / 10% 평가 (자동 분할)

### B. Generation 모델 (mT5)
상황에 맞는 **넛지 내용**을 생성하는 시퀀스-투-시퀀스 모델을 학습합니다.

```bash
python src/train_mt5.py --exp_name my_experiment_v1
```
- **입력**: 사용자 맥락 (Context)
- **출력**: 자연어 제안 (Nudge)
- **로그 저장**: 학습 로그는 `./logs/my_experiment_v1` 에 저장됩니다.

---

## 3. 평가 방법 (Evaluation)

### 자동 분석 (Automated Analysis)
학습이 완료되면 각 모델에 대해 상세 분석 리포트가 자동으로 생성됩니다.

- **리포트 경로**:
    - Trigger 모델: `reports/my_experiment_v1_trigger_analysis.md`
    - Generator 모델: `reports/my_experiment_v1_generation_analysis.md`
- **내용**:
    - **Metrics**: Accuracy, Loss 등 학습 지표
    - **샘플 테이블**: 평가 데이터셋에 대해 모델의 예측 결과(Trigger 여부 또는 생성된 문구)를 실제 정답과 비교한 테이블을 제공합니다.

### 추론 데모 (Inference Demo)
전체 파이프라인(Trigger + Generator)을 대화형으로 테스트하려면 다음 명령어를 실행하세요:

```bash
python src/inference.py --exp_name my_experiment_v1
```

이 스크립트는 `models/.../my_experiment_v1`에 저장된 학습된 모델을 로드하여 샘플 맥락에 대한 예측을 수행합니다.

<br>
<br>
<br>

---

# English Version

This document explains the datasets used and how to train/evaluate the models.

---

## 1. Datasets

We support multiple conversational datasets to train the Proactive Nudge Engine. The system automatically detects and loads the available dataset in the following priority:

1.  **DailyDialog** (Default/Recommended)
    - High-quality daily conversation data.
2.  **DialogSum** (Fallback)
    - Real-life dialogue summarization dataset.
3.  **SAMsum** (Fallback)
    - Messenger-like conversations.

If no real dataset is found, the system generates **Synthetic Data** (1,000+ samples) automatically.

**Download Command:**
```bash
python src/download_data.py
```

---

## 2. Training

You can train the Trigger (BERT) and Generator (mT5) models separately. Use the `--exp_name` argument to organize your experiments.

### A. Trigger Model (BERT)
Trains a binary classifier to decide **"Is a nudge needed?"**

```bash
python src/train_bert.py --exp_name my_experiment_v1
```
- **Input**: User context
- **Output**: 1 (Need Nudge) / 0 (No Nudge)
- **Data Split**: 90% Train / 10% Eval (Automatically split)

### B. Generation Model (mT5)
Trains a sequence-to-sequence model to generate the **nudge content**.

```bash
python src/train_mt5.py --exp_name my_experiment_v1
```
- **Input**: User context
- **Output**: Natural language suggestion
- **Logging**: Training logs are saved to `./logs/my_experiment_v1`

---

## 3. Evaluation

### Automated Analysis
Detailed analysis reports are automatically generated for each model after training.

- **Report Path**:
    - Trigger Model: `reports/my_experiment_v1_trigger_analysis.md`
    - Generator Model: `reports/my_experiment_v1_generation_analysis.md`
- **Content**:
    - **Metrics**: Accuracy, Loss, and other training indicators.
    - **Sample Table**: Comparison tables showing the model's predictions (Trigger decision or Generated text) against the ground truth.

### Inference Demo
To test the full pipeline (Trigger + Generator) interactively:

```bash
python src/inference.py --exp_name my_experiment_v1
```

This script loads the trained models from `models/.../my_experiment_v1` and runs prediction on sample contexts.
