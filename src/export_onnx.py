import torch
import argparse
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def export_to_onnx(exp_name):
    model_path = f"models/bert_trigger/{exp_name}"
    output_path = f"src/mobilebert_trigger.onnx"
    
    print(f"Loading model from {model_path}...")
    try:
        # Load slow tokenizer to ensure save_vocab works smoothly
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    # Create dummy input for export tracing
    dummy_text = "This is a sample input for export."
    inputs = tokenizer(dummy_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Exporting to ONNX at {output_path}...")
    
    # Export
    # Dynamo-based export in newer PyTorch versions can be unstable with dynamic_axes
    # We switch to standard torch.onnx.export (legacy/script) for better compatibility
    torch.onnx.export(
        model, 
        (inputs['input_ids'], inputs['attention_mask']), 
        output_path,
        input_names=['input_ids', 'attention_mask'], 
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        opset_version=18, # Updated to 18 to resolve PyTorch 2.x export issue
        do_constant_folding=True,
    )
    
    # Also save vocab.txt for the tokenizer in Android
    vocab_path = f"src/vocab.txt"
    # MobileBertTokenizerFast does not support save_vocab directly in the same way
    # We use save_pretrained to save all tokenizer files (vocab.txt, tokenizer.json, etc.)
    tokenizer.save_pretrained(os.path.dirname(vocab_path))
    print(f"Tokenizer files saved to {os.path.dirname(vocab_path)}")
    print("Export complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="magic_cue_v1", help="Experiment name")
    args = parser.parse_args()
    
    export_to_onnx(args.exp_name)
