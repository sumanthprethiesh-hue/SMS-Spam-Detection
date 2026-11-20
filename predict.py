"""Classify one SMS message with a trained model."""

import argparse
from pathlib import Path

import joblib


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether an SMS is spam or ham.")
    parser.add_argument("message", help="SMS text to classify")
    parser.add_argument("--model", default="models/sms_spam_model.joblib", help="Saved model path")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        parser.error(f"Model not found at {model_path}. Train it first with: python train.py")

    model = joblib.load(model_path)
    label = model.predict([args.message])[0]
    if label not in {"spam", "ham"}:
        parser.error(f"Model returned an unexpected label: {label!r}")

    if label == "spam":
        print("Result: This SMS is SPAM.")
    else:
        print("Result: This SMS is NOT SPAM.")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([args.message])[0]
        class_probabilities = dict(zip(model.classes_, probabilities))
        spam_probability = class_probabilities.get("spam")
        if spam_probability is not None:
            print(f"Spam probability: {spam_probability:.1%}")


if __name__ == "__main__":
    main()
