import argparse
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

class ProactiveNudgeEngine:
    def __init__(self, trigger_model_path="models/bert_trigger", generator_model_path="models/flan_t5_generator"):
        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Using device: {self.device}")

        # Load Trigger Model
        print(f"Loading Trigger Model from {trigger_model_path}...")
        try:
            self.trigger_tokenizer = AutoTokenizer.from_pretrained(trigger_model_path)
            self.trigger_model = AutoModelForSequenceClassification.from_pretrained(trigger_model_path).to(self.device)
        except OSError:
            print(f"Warning: Could not load trained model from {trigger_model_path}. Loading base model 'bert-base-uncased' instead.")
            self.trigger_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.trigger_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(self.device)

        # Load Generator Model
        print(f"Loading Generator Model from {generator_model_path}...")
        try:
            self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_path)
            self.generator_model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_path).to(self.device)
        except OSError:
            print(f"Warning: Could not load trained model from {generator_model_path}. Loading base model 'google/mt5-small' instead.")
            self.generator_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
            self.generator_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small").to(self.device)

    def predict(self, context):
        # Step 1: Trigger Check
        inputs = self.trigger_tokenizer(
            context, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.trigger_model(**inputs)
            print('outputs.logits:', outputs.logits)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            print('probabilities:', probabilities)
            prediction = torch.argmax(probabilities, dim=-1).item()
            print('prediction:', prediction)
            
        #if prediction == 0:
         #   return {
          #      "context": context,
           #     "trigger": False,
            #    "nudge": None,
             #   "confidence": probabilities[0][0].item()
            #}
        
        
        # Step 2: Generation (if trigger is True)
        # Add task prefix for T5
        gen_inputs = self.generator_tokenizer(
            "nudge: " + context,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.generator_model.generate(
                **gen_inputs, 
                max_length=50, 
                num_beams=4, 
                early_stopping=True
            )
            
        nudge_text = self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "context": context,
            "trigger": True,
            "nudge": nudge_text,
            "confidence": probabilities[0][1].item()
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="default", help="Name of the experiment to load")
    args = parser.parse_args()
    
    # Construct paths based on exp_name
    trigger_path = f"models/bert_trigger/{args.exp_name}"
    generator_path = f"models/flan_t5_generator/{args.exp_name}"
    
    engine = ProactiveNudgeEngine(trigger_model_path=trigger_path, generator_model_path=generator_path)
    
    test_contexts = [
        "I have a meeting with the client at 3 PM.",
        "It's sunny and warm outside.",
        "My flight leaves in 2 hours and I haven't packed.",
        "Just watching a movie.",
        "# person1 # : are you sure? # person2 # : i know it does. i take this bus a lot. # person1 # : how long does the bus take to get there?"
    ]
    
    print("\n--- Running Inference Tests ---\n")
    for ctx in test_contexts:
        result = engine.predict(ctx)
        print(f"Context: {result['context']}")
        print(f"Trigger: {result['trigger']} (Conf: {result['confidence']:.2f})")
        if result['trigger']:
            print(f"Nudge: {result['nudge']}")
        print("-" * 30)

