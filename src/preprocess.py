import torch
from torch.utils.data import Dataset
import random

class ProactiveDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128, task="trigger"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task # 'trigger' or 'generation'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        
        # Tokenize context
        inputs = self.tokenizer(
            context,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()

        if self.task == "trigger":
            label = torch.tensor(item['trigger_label'], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": label
            }
        
        elif self.task == "generation":
            target = item['target_nudge']
            # Tokenize target
            labels = self.tokenizer(
                text_target=target,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze()
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

def create_dummy_data(num_samples=100):
    """
    Creates dummy data for testing the pipeline without external datasets.
    """
    data = []
    contexts = [
        "I have a meeting with John at 2 PM.",
        "The weather looks cloudy today.",
        "My flight is at 6 PM, traffic is heavy.",
        "I'm just sitting on the couch watching TV.",
        "It's my mom's birthday tomorrow.",
        "I need to buy groceries.",
        "Just finished lunch.",
        "Walking in the park."
    ]
    
    # Simple rule-based dummy generation
    for _ in range(num_samples):
        ctx = random.choice(contexts)
        if "meeting" in ctx:
            label = 1
            nudge = "Prepare the presentation slides."
        elif "cloudy" in ctx:
            label = 1
            nudge = "Bring an umbrella just in case."
        elif "flight" in ctx:
            label = 1
            nudge = "Leave 30 minutes earlier to avoid traffic."
        elif "birthday" in ctx:
            label = 1
            nudge = "Order a cake and buy a gift."
        elif "groceries" in ctx:
            label = 1
            nudge = "Check the fridge before you go."
        else:
            label = 0
            nudge = "" # No nudge needed
            
        data.append({
            "context": ctx,
            "trigger_label": label,
            "target_nudge": nudge
        })
    
    return data

def load_processed_data(tokenizer, task="trigger", num_samples=100):
    raw_data = create_dummy_data(num_samples)
    if task == "generation":
        # Filter only positive examples for generation training
        raw_data = [d for d in raw_data if d['trigger_label'] == 1]
        
    dataset = ProactiveDataset(raw_data, tokenizer, task=task)
    return dataset
