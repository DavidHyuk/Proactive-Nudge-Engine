import torch
import os
import random
from torch.utils.data import Dataset
from datasets import load_from_disk

class ProactiveDataset(Dataset):
    def __init__(self, tokenized_inputs, tokenized_targets=None, labels=None, task="trigger"):
        self.input_ids = tokenized_inputs['input_ids']
        self.attention_mask = tokenized_inputs['attention_mask']
        self.task = task
        
        # For trigger task, labels are simple integers
        if task == "trigger":
            self.labels = labels
        # For generation task, labels are tokenized target sequences
        elif task == "generation":
            self.labels = tokenized_targets['input_ids']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        
        if self.task == "trigger":
            item["labels"] = self.labels[idx]
        elif self.task == "generation":
            # Return as list to avoid 'list of numpy arrays' warning in DataCollator
            item["labels"] = self.labels[idx].tolist()
            
        return item

def create_dummy_data(num_samples=1000):
    """
    Creates synthetic data dictionary.
    """
    data = []
    
    positives = [
        ("I have a meeting with [PERSON] at [TIME].", "Prepare the presentation slides and review notes."),
        ("It looks like it's going to rain today.", "Don't forget to take an umbrella."),
        ("My flight leaves at [TIME], and traffic is bad.", "Leave 30 minutes earlier to avoid traffic."),
        ("It's [PERSON]'s birthday tomorrow.", "Order a cake and buy a gift."),
        ("I need to buy groceries for the week.", "Check the fridge and make a shopping list."),
        ("I feel like I'm gaining weight.", "Consider going to the gym or for a run."),
        ("I haven't called [PERSON] in a while.", "Give them a call to catch up."),
        ("The house is getting messy.", "Schedule a cleaning session for this weekend."),
        ("I'm running low on gas.", "Stop by the gas station on your way out."),
        ("I have a dentist appointment tomorrow.", "Remember to brush and floss extra well tonight.")
    ]
    
    negatives = [
        "I'm just watching TV.",
        "The sky is blue today.",
        "I had a good lunch.",
        "Walking the dog in the park.",
        "Reading a book on the sofa.",
        "Just finished work.",
        "Listening to music.",
        "Sleeping in today.",
        "Playing video games.",
        "Chatting with a friend about nothing."
    ]
    
    people = ["John", "Sarah", "Mom", "Dad", "the boss", "Alice"]
    times = ["2 PM", "5 PM", "9 AM", "noon", "6:30 PM"]
    
    for _ in range(num_samples):
        is_trigger = random.choice([True, False])
        
        if is_trigger:
            template, nudge_template = random.choice(positives)
            ctx = template.replace("[PERSON]", random.choice(people)).replace("[TIME]", random.choice(times))
            nudge = nudge_template 
            data.append({
                "context": ctx,
                "trigger_label": 1,
                "target_nudge": nudge
            })
        else:
            ctx = random.choice(negatives)
            if random.random() > 0.5:
                ctx = ctx.replace(".", " and relaxing.")
            data.append({
                "context": ctx,
                "trigger_label": 0,
                "target_nudge": ""
            })
            
    return data

def tokenize_data(data, tokenizer, task="trigger", max_length=128):
    """
    Helper to tokenize a list of dicts.
    """
    # For T5, adding a task prefix is standard practice
    if task == "generation":
        contexts = ["nudge: " + d['context'] for d in data]
    else:
        contexts = [d['context'] for d in data]
    
    # Tokenize inputs in batch (Fast!)
    tokenized_inputs = tokenizer(
        contexts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    if task == "trigger":
        import numpy as np
        labels_list = [d['trigger_label'] for d in data]
        # Pre-convert to tensor via numpy for efficiency and to resolve warning
        labels = torch.tensor(np.array(labels_list), dtype=torch.long)
        return ProactiveDataset(tokenized_inputs, labels=labels, task=task)
    
    elif task == "generation":
        targets = [d['target_nudge'] for d in data]
        # Tokenize targets in batch using text_target
        tokenized_targets = tokenizer(
            text_target=targets,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # IMPORTANT: Replace pad_token_id with -100 so it's ignored by the loss function
        labels = tokenized_targets['input_ids'].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        tokenized_targets['input_ids'] = labels
        
        return ProactiveDataset(tokenized_inputs, tokenized_targets=tokenized_targets, task=task)

def load_real_data(tokenizer, task="trigger", split="train"):
    raw_data = []
    
    # Try DailyDialog
    if os.path.exists("data/dailydialog"):
        print(f"Loading DailyDialog ({split})...")
        try:
            dataset = load_from_disk("data/dailydialog")
            full_data = dataset['train']
            
            # 90% train, 10% eval
            split_idx = int(len(full_data) * 0.9)
            if split == "train":
                data_subset = full_data.select(range(0, split_idx))
            else:
                data_subset = full_data.select(range(split_idx, len(full_data)))

            for i in range(len(data_subset)):
                item = data_subset[i]
                dialog = item['dialog']
                # Use a window of history
                history_window = 3
                for j in range(len(dialog) - 1):
                    # Construct context from up to 'history_window' previous turns + current turn
                    start_idx = max(0, j - history_window + 1)
                    context_turns = dialog[start_idx : j+1]
                    # Join with a separator (e.g., newline or special token)
                    context_str = " ".join(context_turns)
                    
                    if len(dialog[j].split()) > 2:
                        raw_data.append({
                            "context": context_str,
                            "trigger_label": 1,
                            "target_nudge": dialog[j+1]
                        })
        except Exception as e:
            print(f"Error loading DailyDialog: {e}")

    # Try DialogSum
    elif os.path.exists("data/dialogsum"):
        print(f"Loading DialogSum ({split})...")
        try:
            dataset = load_from_disk("data/dialogsum")
            full_data = dataset['train']
            
            split_idx = int(len(full_data) * 0.9)
            if split == "train":
                data_subset = full_data.select(range(0, split_idx))
            else:
                data_subset = full_data.select(range(split_idx, len(full_data)))

            for i in range(len(data_subset)):
                item = data_subset[i]
                lines = item['dialogue'].split('\n')
                history_window = 3
                
                for j in range(len(lines) - 1):
                    # Get target
                    p2 = lines[j+1].split(':', 1)
                    
                    if len(p2) == 2:
                        # Construct context
                        start_idx = max(0, j - history_window + 1)
                        context_lines = lines[start_idx : j+1]
                        context_str = " ".join(context_lines) # Keep Speaker: Text format
                        
                        raw_data.append({
                            "context": context_str,
                            "trigger_label": 1,
                            "target_nudge": p2[1].strip()
                        })
        except Exception as e:
            print(f"Error loading DialogSum: {e}")

    # Try SAMsum
    elif os.path.exists("data/samsum"):
        print(f"Loading SAMsum ({split})...")
        try:
            dataset = load_from_disk("data/samsum")
            full_data = dataset['train']
            
            split_idx = int(len(full_data) * 0.9)
            if split == "train":
                data_subset = full_data.select(range(0, split_idx))
            else:
                data_subset = full_data.select(range(split_idx, len(full_data)))

            for i in range(len(data_subset)):
                item = data_subset[i]
                lines = item['dialogue'].split('\n')
                history_window = 3
                
                for j in range(len(lines) - 1):
                    p2 = lines[j+1].split(':', 1)
                    
                    if len(p2) == 2:
                        start_idx = max(0, j - history_window + 1)
                        context_lines = lines[start_idx : j+1]
                        context_str = " ".join(context_lines)
                        
                        raw_data.append({
                            "context": context_str,
                            "trigger_label": 1,
                            "target_nudge": p2[1].strip()
                        })
        except Exception as e:
            print(f"Error loading SAMsum: {e}")

    if not raw_data:
        return None
        
    # Add negatives for Trigger task
    if task == "trigger":
        neg = create_dummy_data(len(raw_data)//4)
        raw_data.extend([d for d in neg if d['trigger_label']==0])
        random.shuffle(raw_data)
        
    print(f"Tokenizing {len(raw_data)} samples... (This might take a moment)")
    return tokenize_data(raw_data, tokenizer, task=task)

def load_processed_data(tokenizer, task="trigger", num_samples=100, use_dummy=False, split="train"):
    if not use_dummy:
        dataset = load_real_data(tokenizer, task, split=split)
        if dataset:
            return dataset
            
    print(f"Using synthetic data generator ({split})...")
    samples_to_gen = num_samples if split == "train" else max(num_samples // 5, 20)
    raw_data = create_dummy_data(samples_to_gen)
    
    if task == "generation":
        raw_data = [d for d in raw_data if d['trigger_label'] == 1]
        
    print(f"Tokenizing {len(raw_data)} synthetic samples...")
    return tokenize_data(raw_data, tokenizer, task=task)
