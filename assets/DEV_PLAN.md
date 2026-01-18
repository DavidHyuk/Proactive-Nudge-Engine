# Development Plan: Proactive-Nudge-Engine

This document outlines the step-by-step development plan for the Proactive-Nudge-Engine.

- [x] **Step 1: Project Initialization & Structure Setup**
    - Create the directory structure: `data/`, `models/`, `src/`, `notebooks/`, `scripts/`.
    - Verify `requirements.txt` matches the environment setup.

- [x] **Step 2: Data Preprocessing Module (`src/preprocess.py`)**
    - Implement a `DataProcessor` class to handle dataset loading (e.g., Taskmaster, MultiWOZ).
    - Create functions to tokenize inputs for BERT (classification) and mT5 (generation).
    - Implement a mock data generator for testing purposes if real datasets are not present.

- [x] **Step 3: Trigger Model Implementation (`src/train_bert.py`)**
    - Define the BERT Classifier architecture for the "Triggering" step.
    - Implement the training loop (training, validation).
    - Save the best model to `models/bert_trigger/`.

- [x] **Step 4: Generation Model Implementation (`src/train_mt5.py`)**
    - Define the mT5 Generator architecture for the "Generation" step.
    - Implement the training loop (seq2seq fine-tuning).
    - Save the best model to `models/mt5_generator/`.

- [x] **Step 5: System Integration & Inference Demo**
    - Create a script (e.g., `src/inference.py`) that chains the two models.
    - Implement the workflow: `Context -> BERT (Check Nudge) -> if True -> mT5 (Generate Nudge)`.
    - Verify with sample inputs.

- [x] **Step 6: Data Verification Notebook**
    - Create `notebooks/01_verify_data.ipynb`.
    - Visualize loaded data and tokenization results.
