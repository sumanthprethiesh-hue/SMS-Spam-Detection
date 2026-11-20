"""Train and evaluate the SMS spam classifier."""

import argparse
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.sms_model import build_model, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an SMS spam prediction model.")
    parser.add_argument("--data", default="data/sms_spam.csv", help="Path to labeled CSV data")
    parser.add_argument("--output", default="models/sms_spam_model.joblib", help="Model output path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction held out for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split")
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")

    messages, labels = load_dataset(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        messages,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    classes = list(model.classes_)

    print(f"Training rows: {len(x_train)} | Test rows: {len(x_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=["ham", "spam"], zero_division=0))

    print("Test SMS predictions:")
    for index, (message, actual, predicted) in enumerate(zip(x_test, y_test, predictions)):
        class_probabilities = dict(zip(classes, probabilities[index]))
        spam_probability = class_probabilities.get("spam", 0.0)
        ham_probability = class_probabilities.get("ham", 0.0)
        result = "SPAM" if predicted == "spam" else "NOT SPAM"
        print(f"\nSMS: {message}")
        print(f"Actual label: {actual}")
        print(f"Prediction: {result}")
        print(f"Spam: {spam_probability:.1%} | Not spam: {ham_probability:.1%}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
