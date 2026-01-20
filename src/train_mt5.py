import os
import torch
import numpy as np
import argparse
import wandb
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
try:
    from preprocess import load_processed_data
except ImportError:
    from src.preprocess import load_processed_data

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="default", help="Name of the experiment")
    args = parser.parse_args()

    model_name = "google/mt5-small"
    output_dir = f"models/mt5_generator/{args.exp_name}"
    logging_dir = f"./logs/{args.exp_name}"
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading dataset...")
    # Only "nudge needed" examples are useful for generation training
    train_dataset = load_processed_data(tokenizer, task="generation", num_samples=100, split="train")
    eval_dataset = load_processed_data(tokenizer, task="generation", num_samples=20, split="eval")
    
    print(f"Loading model: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Initialize WandB
    wandb.init(
        project="Proactive-Nudge-Engine",
        name=f"mt5-generator-{args.exp_name}",
        config={
            "model": model_name,
            "experiment": args.exp_name,
            "task": "generation"
        }
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        learning_rate=3e-4,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir=logging_dir,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        predict_with_generate=True,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        report_to="wandb",
        use_cpu=not torch.cuda.is_available()
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Generation Analysis Step
    print("Running generation analysis on evaluation set...")
    predictions, labels, metrics = trainer.predict(eval_dataset, metric_key_prefix="predict")
    
    # Decode predictions and labels
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in labels as we can't decode them
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Decode inputs
    # Need to access raw input_ids from eval_dataset items
    decoded_inputs = []
    for i in range(len(eval_dataset)):
        item = eval_dataset[i]
        decoded_inputs.append(tokenizer.decode(item['input_ids'], skip_special_tokens=True))

    # Save to reports
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/{args.exp_name}_generation_analysis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Generation Analysis: {args.exp_name}\n\n")
        f.write(f"## Metrics\n\n")
        for k, v in metrics.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Sample Generations\n\n")
        f.write("| Context | Reference (Target) | Generated Nudge |\n")
        f.write("| :--- | :--- | :--- |\n")
        # Save first 50 samples
        for i in range(min(50, len(decoded_preds))):
            ctx = decoded_inputs[i].replace("\n", " ")
            ref = decoded_labels[i].replace("\n", " ")
            pred = decoded_preds[i].replace("\n", " ")
            f.write(f"| {ctx} | {ref} | {pred} |\n")
    
    print(f"Analysis report saved to {report_path}")

if __name__ == "__main__":
    train()
