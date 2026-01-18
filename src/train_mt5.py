import os
import torch
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
    model_name = "google/mt5-small"
    output_dir = "models/mt5_generator"
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading dataset...")
    # Only "nudge needed" examples are useful for generation training
    train_dataset = load_processed_data(tokenizer, task="generation", num_samples=100)
    eval_dataset = load_processed_data(tokenizer, task="generation", num_samples=20)
    
    print(f"Loading model: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
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

if __name__ == "__main__":
    train()
