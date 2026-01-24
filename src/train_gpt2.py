import os
import torch
import numpy as np
import argparse
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
try:
    from preprocess import load_processed_data
except ImportError:
    from src.preprocess import load_processed_data

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="default", help="Name of the experiment")
    args = parser.parse_args()

    model_name = "distilgpt2"
    output_dir = f"models/gpt2_generator/{args.exp_name}"
    logging_dir = f"./logs/{args.exp_name}"
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading dataset...")
    # Only "nudge needed" examples are useful for generation training
    train_dataset = load_processed_data(tokenizer, task="generation-causal", num_samples=1000, split="train")
    eval_dataset = load_processed_data(tokenizer, task="generation-causal", num_samples=200, split="eval")
    
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Initialize WandB
    wandb.init(
        project="Proactive-Nudge-Engine",
        name=f"gpt2-generator-{args.exp_name}",
        config={
            "model": model_name,
            "experiment": args.exp_name,
            "task": "generation-causal"
        }
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1, 
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
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
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Generation Analysis Step
    print("Running generation analysis on evaluation set...")
    # Get metrics (loss, etc.)
    predictions_output = trainer.predict(eval_dataset, metric_key_prefix="predict")
    metrics = predictions_output.metrics

    # Manual generation for analysis
    print("Generating samples...")
    model.eval()
    decoded_preds = []
    decoded_labels = []
    decoded_inputs = []
    
    # Analyze first 50 samples
    for i in range(min(50, len(eval_dataset))):
        item = eval_dataset[i]
        # Ensure input_ids is a tensor and on the correct device
        input_ids = torch.tensor([item['input_ids']], device=model.device) if not torch.is_tensor(item['input_ids']) else item['input_ids'].unsqueeze(0).to(model.device)
        
        # Decode to text to split context and target
        full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        if "\nNUDGE:\n" in full_text:
            context_part = full_text.split("\nNUDGE:\n")[0]
            prompt_text = context_part + "\nNUDGE:\n"
            target_nudge = full_text.split("\nNUDGE:\n")[1]
        else:
            context_part = full_text
            prompt_text = full_text
            target_nudge = ""
            
        decoded_inputs.append(context_part)
        decoded_labels.append(target_nudge)
        
        # Generate
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=4,
                early_stopping=True
            )
        
        generated_full = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        # Extract the new part
        if "\nNUDGE:\n" in generated_full:
            generated_nudge = generated_full.split("\nNUDGE:\n")[1]
        else:
            # If the model didn't reproduce the prompt exactly or something went wrong, just take the suffix
            generated_nudge = generated_full[len(prompt_text):]
            
        decoded_preds.append(generated_nudge)


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
        
        for i in range(len(decoded_preds)):
            ctx = decoded_inputs[i].replace("\n", " ")
            ref = decoded_labels[i].replace("\n", " ")
            pred = decoded_preds[i].replace("\n", " ")
            f.write(f"| {ctx} | {ref} | {pred} |\n")
    
    print(f"Analysis report saved to {report_path}")

if __name__ == "__main__":
    train()
