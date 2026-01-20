import os
import torch
import argparse
import wandb
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
        num_train_epochs=1,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir=logging_dir,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        report_to="wandb",
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
            gt = "Trigger" if labels[i] == 1 else "No Trigger"
            pr = "Trigger" if preds[i] == 1 else "No Trigger"
            res = "✅" if labels[i] == preds[i] else "❌"
            f.write(f"| {ctx} | {gt} | {pr} | {res} |\n")
    
    print(f"Analysis report saved to {report_path}")

if __name__ == "__main__":
    train()
