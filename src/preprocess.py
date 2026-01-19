import torch
import os
import random
from torch.utils.data import Dataset
from datasets import load_from_disk

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

def create_dummy_data(num_samples=1000):
    """
    Creates a large synthetic dataset to mimic real data behavior.
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

def load_real_data(tokenizer, task="trigger"):
    # Try DailyDialog
    if os.path.exists("data/dailydialog"):
        print("Loading DailyDialog...")
        try:
            dataset = load_from_disk("data/dailydialog")
            data_split = dataset['train']
            processed_data = []
            for i in range(min(len(data_split), 5000)):
                item = data_split[i]
                dialog = item['dialog']
                for j in range(len(dialog) - 1):
                    if len(dialog[j].split()) > 2:
                        processed_data.append({
                            "context": dialog[j],
                            "trigger_label": 1,
                            "target_nudge": dialog[j+1]
                        })
            
            if task == "trigger":
                neg = create_dummy_data(len(processed_data)//4)
                processed_data.extend([d for d in neg if d['trigger_label']==0])
                random.shuffle(processed_data)
            
            print(f"Loaded {len(processed_data)} samples from DailyDialog.")
            return ProactiveDataset(processed_data, tokenizer, task=task)
        except Exception as e:
            print(f"Error loading DailyDialog: {e}")

    # Try DialogSum (The fallback added by user)
    if os.path.exists("data/dialogsum"):
        print("Loading DialogSum...")
        try:
            dataset = load_from_disk("data/dialogsum")
            data_split = dataset['train']
            processed_data = []
            for i in range(min(len(data_split), 5000)):
                item = data_split[i]
                lines = item['dialogue'].split('\n')
                for j in range(len(lines) - 1):
                    p1 = lines[j].split(':', 1)
                    p2 = lines[j+1].split(':', 1)
                    if len(p1)==2 and len(p2)==2:
                        processed_data.append({
                            "context": p1[1].strip(),
                            "trigger_label": 1,
                            "target_nudge": p2[1].strip()
                        })
            
            if task == "trigger":
                neg = create_dummy_data(len(processed_data)//4)
                processed_data.extend([d for d in neg if d['trigger_label']==0])
                random.shuffle(processed_data)
            
            print(f"Loaded {len(processed_data)} samples from DialogSum.")
            return ProactiveDataset(processed_data, tokenizer, task=task)
        except Exception as e:
            print(f"Error loading DialogSum: {e}")

    # Try SAMsum
    if os.path.exists("data/samsum"):
        print("Loading SAMsum...")
        try:
            dataset = load_from_disk("data/samsum")
            data_split = dataset['train']
            processed_data = []
            for i in range(min(len(data_split), 5000)):
                item = data_split[i]
                lines = item['dialogue'].split('\n')
                for j in range(len(lines) - 1):
                    p1 = lines[j].split(':', 1)
                    p2 = lines[j+1].split(':', 1)
                    if len(p1)==2 and len(p2)==2:
                        processed_data.append({
                            "context": p1[1].strip(),
                            "trigger_label": 1,
                            "target_nudge": p2[1].strip()
                        })
            
            if task == "trigger":
                neg = create_dummy_data(len(processed_data)//4)
                processed_data.extend([d for d in neg if d['trigger_label']==0])
                random.shuffle(processed_data)
            
            print(f"Loaded {len(processed_data)} samples from SAMsum.")
            return ProactiveDataset(processed_data, tokenizer, task=task)
        except Exception as e:
            print(f"Error loading SAMsum: {e}")

    return None

def load_processed_data(tokenizer, task="trigger", num_samples=100, use_dummy=False):
    if not use_dummy:
        dataset = load_real_data(tokenizer, task)
        if dataset:
            return dataset
            
    print("Using synthetic data generator...")
    raw_data = create_dummy_data(1000 if num_samples < 1000 else num_samples)
    if task == "generation":
        raw_data = [d for d in raw_data if d['trigger_label'] == 1]
    return ProactiveDataset(raw_data, tokenizer, task=task)
