import os
import torch
import argparse
import wandb
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
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
    
    model_name = "google/mobilebert-uncased" # Using MobileBERT for smaller size
    output_dir = f"models/bert_trigger/{args.exp_name}"
    logging_dir = f"./logs/{args.exp_name}"
    
    print(f"Loading tokenizer: {model_name}")
    # Use use_fast=False to ensure vocab.txt is saved correctly for MobileBERT
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    print("Loading dataset...")
    # Load Real Data (use_dummy=False)
    # This will load DailyDialog/DialogSum/SAMsum if available and auto-label them
    train_dataset = load_processed_data(tokenizer, task="trigger", num_samples=1000, split="train", use_dummy=False)
    
    # If real data failed to load (e.g. dataset not downloaded), fallback to dummy but warn user
    if train_dataset is None:
        print(">>> WARNING: Real data not found or failed to load. Falling back to SYNTHETIC (DUMMY) data for training. <<<")
        train_dataset = load_processed_data(tokenizer, task="trigger", num_samples=1000, split="train", use_dummy=True)
    else:
        print(">>> SUCCESS: Real data loaded successfully for training. <<<")

    eval_dataset = load_processed_data(tokenizer, task="trigger", num_samples=200, split="eval", use_dummy=False)
    if eval_dataset is None:
         print(">>> WARNING: Real data not found for evaluation. Falling back to SYNTHETIC (DUMMY) data. <<<")
         eval_dataset = load_processed_data(tokenizer, task="trigger", num_samples=200, split="eval", use_dummy=True)
    else:
         print(">>> SUCCESS: Real data loaded successfully for evaluation. <<<")
    
    print(f"Loading model: {model_name}")
    # Labels: 0=None, 1=Passport, 2=Wifi, 3=Address, 4=Schedule, 5=Flight, 6=Membership
    num_labels = 7 
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels 
    )

    # Initialize WandB
    wandb.init(
        project="Proactive-Nudge-Engine",
        name=f"bert-trigger-{args.exp_name}",
        config={
            "model": model_name,
            "experiment": args.exp_name,
            "task": "trigger"
        }
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        warmup_steps=20,
        weight_decay=0.01,
        logging_dir=logging_dir,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="wandb",
        use_cpu=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    # Important: Save tokenizer explicitly using save_pretrained
    tokenizer.save_pretrained(output_dir)

    # Classification Analysis Step
    print("Running classification analysis on evaluation set...")
    predictions_output = trainer.predict(eval_dataset)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    metrics = predictions_output.metrics
    
    # Get predicted classes
    import numpy as np
    preds = np.argmax(logits, axis=-1)
    
    # Decode inputs
    decoded_inputs = []
    for i in range(len(eval_dataset)):
        item = eval_dataset[i]
        decoded_inputs.append(tokenizer.decode(item['input_ids'], skip_special_tokens=True))

    # Save to reports
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/{args.exp_name}_trigger_analysis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Trigger Classification Analysis: {args.exp_name}\n\n")
        f.write(f"## Metrics\n\n")
        for k, v in metrics.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Sample Predictions\n\n")
        f.write("| Context | Ground Truth | Prediction | Result |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        # Save first 50 samples
        for i in range(min(50, len(preds))):
            ctx = decoded_inputs[i].replace("\n", " ")
            gt = str(labels[i])
            pr = str(preds[i])
            res = "✅" if labels[i] == preds[i] else "❌"
            f.write(f"| {ctx} | {gt} | {pr} | {res} |\n")
    
    print(f"Analysis report saved to {report_path}")

if __name__ == "__main__":
    train()
