import os
import torch
import argparse
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
# Assuming execution from project root via 'python src/train_bert.py' might add src to path,
# or simply 'python src/train_bert.py' makes src the script dir.
try:
    from preprocess import load_processed_data
except ImportError:
    from src.preprocess import load_processed_data

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="default", help="Name of the experiment")
    args = parser.parse_args()
    
    model_name = "bert-base-uncased" # Using standard BERT as trigger
    output_dir = f"models/bert_trigger/{args.exp_name}"
    logging_dir = f"./logs/{args.exp_name}"
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading dataset...")
    # Generate sufficient dummy data for training
    train_dataset = load_processed_data(tokenizer, task="trigger", num_samples=100, split="train")
    eval_dataset = load_processed_data(tokenizer, task="trigger", num_samples=20, split="eval")
    
    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2 # 0: No Nudge, 1: Nudge
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir=logging_dir,
        logging_steps=10,
        eval_strategy="epoch", # newer version preference
        save_strategy="epoch",
        load_best_model_at_end=True,
        use_cpu=not torch.cuda.is_available()
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train()
