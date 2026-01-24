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
        elif task == "generation-causal":
            self.labels = labels

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
    Creates synthetic data dictionary with specific category labels.
    Labels:
    0: No Trigger
    1: Passport
    2: Wifi
    3: Address
    4: Schedule (Meeting/Appointment)
    5: Weather
    6: Flight
    7: Membership
    """
    data = []
    
    # Define templates by category
    categories = {
        1: [ # Passport
            ("I need to book a flight but I don't have my passport info handy.", "Show Passport #A12345678"),
            ("Can you send me your passport details for the reservation?", "Show Passport #A12345678"),
            ("I'm filling out the visa form.", "Show Passport #A12345678"),
            ("Do you have your passport number?", "Show Passport #A12345678"),
            ("I need to enter my passport expiry date for the check-in.", "Show Passport #A12345678"),
            ("The travel agent is asking for a copy of my passport.", "Show Passport #A12345678"),
            ("I left my passport at home, do you have a picture of it?", "Show Passport #A12345678"),
            ("Filling out the immigration card, what's my document number?", "Show Passport #A12345678"),
            ("Need passport details for the hotel registration.", "Show Passport #A12345678"),
            ("Booking an international train, passport required.", "Show Passport #A12345678"),
        ],
        2: [ # Wifi
            ("What's the wifi password again?", "Show WiFi: MyNetwork / Pass123"),
            ("I need to connect to the internet.", "Show WiFi: Guest_Wifi / guest123"),
            ("Can I get on your wifi?", "Show WiFi: MyNetwork / Pass123"),
            ("My laptop isn't connecting, what was the network key?", "Show WiFi: MyNetwork / Pass123"),
            ("Is there free wifi here? What's the login?", "Show WiFi: Guest_Wifi / guest123"),
            ("Hey, can you share the wifi credentials?", "Show WiFi: MyNetwork / Pass123"),
            ("I ran out of data, need to use the WLAN.", "Show WiFi: MyNetwork / Pass123"),
            ("Setting up the new device, need wifi pass.", "Show WiFi: MyNetwork / Pass123"),
            ("The guest wifi password is needed for the visitors.", "Show WiFi: Guest_Wifi / guest123"),
            ("Connecting the TV to the internet.", "Show WiFi: MyNetwork / Pass123"),
        ],
        3: [ # Address
            ("What's your address?", "Share: 123 Maple St, Springfield"),
            ("I'm coming over, send me the location.", "Share: 456 Oak Ave, Metropolis"),
            ("Where do you live?", "Share: 123 Maple St, Springfield"),
            ("I need to ship this package to you, what's the address?", "Share: 123 Maple St, Springfield"),
            ("Ordering food delivery, need the drop-off location.", "Share: 456 Oak Ave, Metropolis"),
            ("Can you text me your home address?", "Share: 123 Maple St, Springfield"),
            ("I'll pick you up, give me the address.", "Share: 456 Oak Ave, Metropolis"),
            ("Updating my contact info, what's your current address?", "Share: 123 Maple St, Springfield"),
            ("Sending a gift, need the mailing address.", "Share: 123 Maple St, Springfield"),
            ("The cab driver needs the destination address.", "Share: 456 Oak Ave, Metropolis"),
        ],
        4: [ # Schedule
            ("I have a dentist appointment with Dr. Smith at 2 PM.", "View: Dr. Smith (Dentist) @ 2 PM"),
            ("Meeting [PERSON] at Starbucks at [TIME].", "View: Starbucks @ [TIME]"),
            ("Don't forget the team sync at 10 AM in Conference Room B.", "View: Conf Room B @ 10 AM"),
            ("Dinner reservation is at 7 PM at The Italian Place.", "View: The Italian Place @ 7 PM"),
            ("When is my next haircut scheduled?", "View: Salon @ 5 PM"),
            ("Do I have any meetings after lunch?", "View: Project Sync @ 1 PM"),
            ("Check my calendar for tomorrow morning.", "View: Breakfast with Client @ 9 AM"),
            ("I think I have a doctor's visit coming up.", "View: Dr. Jones @ 3 PM"),
            ("What time is the concert tonight?", "View: Concert Hall @ 8 PM"),
            ("Schedule a reminder for the parent-teacher conference.", "View: School @ 4 PM"),
            ("Meeting with the lawyer at [TIME].", "View: Law Office @ [TIME]"),
            ("Lunch date with [PERSON] at noon.", "View: Bistro @ 12 PM"),
        ],
        5: [ # Weather
            ("It looks like it's going to rain today.", "Reminder: Take an umbrella"),
            ("Is it cold outside?", "Reminder: Wear a jacket"),
            ("Checking the forecast for the weekend.", "Reminder: Check Weather App"),
            ("Do I need sunscreen today?", "Reminder: High UV Alert"),
            ("Is it snowing?", "Reminder: Wear boots"),
            ("What's the temperature right now?", "Reminder: Check Thermostat"),
            ("Storm warning in effect.", "Reminder: Close windows"),
            ("It's extremely hot out there.", "Reminder: Drink water"),
            ("Planning a picnic, hope it doesn't rain.", "Reminder: Check Forecast"),
            ("Windy day today.", "Reminder: Secure loose items"),
        ],
        6: [ # Flight
            ("My flight leaves at [TIME], and traffic is bad.", "Alert: Leave 30 mins early"),
            ("What is your flight number?", "Show Flight: UA123"),
            ("I need to check in for my flight.", "Show Flight: UA123"),
            ("When does the plane land?", "Show Flight: UA123"),
            ("Heading to the airport now.", "Show Flight: UA123"),
            ("Is the flight on time?", "Show Flight: UA123"),
            ("Terminal info for my trip to New York.", "Show Flight: UA123"),
            ("Boarding pass for flight UA123.", "Show Flight: UA123"),
            ("Picking up mom from the airport, what's her flight?", "Show Flight: KE081"),
            ("Gate change for the flight.", "Show Flight: UA123"),
        ],
        7: [ # Membership
            ("Do you have your loyalty card for the grocery store?", "Show Loyalty Card: #8839201"),
            ("I need my gym member ID.", "Show Gym ID: #GYM-9922"),
            ("Scanning my point card at the checkout.", "Show Loyalty Card: #8839201"),
            ("What's my library card number?", "Show Library Card: #LIB-1122"),
            ("Discount code for the members?", "Show Member ID: #MEM-5566"),
            ("Costco membership card please.", "Show Costco ID: #111222333"),
            ("Showing my digital ID at the entrance.", "Show Member ID: #MEM-5566"),
            ("Accumulating points for this purchase.", "Show Loyalty Card: #8839201"),
            ("Login requires membership number.", "Show Member ID: #MEM-5566"),
            ("Where is my frequent flyer number?", "Show FF #: #FLY-9988"),
        ]
    }
    
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
        "Chatting with a friend about nothing.",
        "Just scrolling through social media.",
        "Thinking about what to cook for dinner.",
        "The car is red.",
        "I like apples.",
        "Did you see that movie?",
        "My phone battery is low.",
        "I need to charge my laptop.",
        "The coffee is hot.",
        "Look at that bird.",
        "I'm tired.",
        "Happy birthday!",
        "Good morning.",
        "See you later.",
        "I don't know.",
        "Maybe next time.",
        "It was nice meeting you.",
        "What are you doing?",
        "How are you?",
        "I am fine.",
        "Let's go for a walk.",
        "The game was exciting.",
        "I bought a new shirt.",
        "My cat is sleeping.",
        "The door is open.",
        "Turn on the lights.",
        "What is the meaning of life?",
        "Math is hard.",
        "I love coding.",
        "Python is a snake.",
        "Data science is cool."
    ]
    
    people = ["John", "Sarah", "Mom", "Dad", "the boss", "Alice"]
    times = ["2 PM", "5 PM", "9 AM", "noon", "6:30 PM"]
    
    for _ in range(num_samples):
        is_trigger = random.choice([True, False])
        
        if is_trigger:
            label = random.choice(list(categories.keys()))
            template, nudge_template = random.choice(categories[label])
            ctx = template.replace("[PERSON]", random.choice(people)).replace("[TIME]", random.choice(times))
            nudge = nudge_template 
            data.append({
                "context": ctx,
                "trigger_label": label, # Now this is a category ID
                "target_nudge": nudge
            })
        else:
            ctx = random.choice(negatives)
            if random.random() > 0.5:
                ctx = ctx.replace(".", " and relaxing.")
            data.append({
                "context": ctx,
                "trigger_label": 0, # 0 = No Trigger
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
    elif task == "generation-causal":
        # Format: Context \nNUDGE:\n Target
        # Use EOS token to signal end of generation
        eos = tokenizer.eos_token if tokenizer.eos_token else ""
        contexts = [f"{d['context']}\nNUDGE:\n{d['target_nudge']}{eos}" for d in data]
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
    elif task == "generation-causal":
        # For causal generation, labels are the same as input_ids
        labels = tokenized_inputs['input_ids'].clone()
        return ProactiveDataset(tokenized_inputs, labels=labels, task=task)

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
