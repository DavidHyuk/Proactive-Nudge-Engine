import argparse
import torch
import os  # Added os
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class ProactiveNudgeEngine:
    def __init__(
        self,
        trigger_model_path="models/bert_trigger",
        # generator_model_path removed
    ):
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
            self.trigger_model = AutoModelForSequenceClassification.from_pretrained(
                trigger_model_path
            ).to(self.device)
        except OSError:
            print(
                f"Warning: Could not load trained model from {trigger_model_path}. Loading base model 'google/mobilebert-uncased' instead."
            )
            self.trigger_tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
            # 8 labels as defined in preprocess.py/train_bert.py
            self.trigger_model = AutoModelForSequenceClassification.from_pretrained(
                "google/mobilebert-uncased", num_labels=8
            ).to(self.device)

        # Magic Cue Dummy Database
        self.DUMMY_DB = {
            1: "Show Passport #A12345678", # Passport
            2: "Show WiFi: MyNetwork / Pass123", # Wifi
            3: "Share: 123 Maple St, Springfield", # Address
            4: "View: Upcoming Appointment @ 2 PM", # Schedule (Generic fallback)
            5: "Reminder: Check Weather / Take Umbrella", # Weather
            6: "Alert: Flight UA123 departs in 3 hours", # Flight
            7: "Show Loyalty Card: #8839201", # Membership
        }
        
        self.LABEL_MAP = {
            0: "No Trigger",
            1: "Passport",
            2: "Wifi",
            3: "Address",
            4: "Schedule",
            5: "Weather",
            6: "Flight",
            7: "Membership"
        }

    def predict(self, context, threshold=0.5):
        inputs = self.trigger_tokenizer(
            context, return_tensors="pt", truncation=True, padding=True, max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.trigger_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        # Get the predicted class index
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_idx].item()
        
        print(f"Predicted Class: {pred_idx} ({self.LABEL_MAP.get(pred_idx, 'Unknown')}) Conf: {confidence:.2f}")

        # 0 is No Trigger
        if pred_idx == 0:
            return {
                "context": context,
                "trigger": False,
                "category": "None",
                "nudge": None,
                "confidence": confidence,
            }

        # Retrieve nudge from dummy DB
        nudge = self.DUMMY_DB.get(pred_idx, "Relevant Info")

        return {
            "context": context,
            "trigger": True,
            "category": self.LABEL_MAP.get(pred_idx, "Unknown"),
            "nudge": nudge,
            "confidence": confidence,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", type=str, default="default", help="Name of the experiment to load"
    )
    args = parser.parse_args()

    # Construct paths based on exp_name
    trigger_path = f"models/bert_trigger/{args.exp_name}"

    engine = ProactiveNudgeEngine(
        trigger_model_path=trigger_path
    )

    test_contexts = [
        "I need to book a flight but I don't have my passport info handy.",
        "Meeting John at Starbucks at 5 PM.",
        "What's the wifi password again?",
        "I'm coming over, send me the location.",
        "Do you have your loyalty card for the grocery store?",
        "It's sunny and warm outside.", # Should be no trigger or simple one
    ]

    print("\n--- Running Inference Tests ---\n")
    for ctx in test_contexts:
        result = engine.predict(ctx)
        print(f"Context: {result['context']}")
        if result["trigger"]:
             print(f"Trigger: YES ({result['category']}) (Conf: {result['confidence']:.2f})")
             print(f"Nudge: {result['nudge']}")
        else:
             print(f"Trigger: NO (Conf: {result['confidence']:.2f})")
        print("-" * 30)
