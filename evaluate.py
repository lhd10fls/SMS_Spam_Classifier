import argparse
import sys

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from spam_classifier.data import load_dataset
from spam_classifier.paths import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH
from spam_classifier.pipeline import load_pipeline


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved spam classifier pipeline.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--data", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--test-size", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_dataset(args.data)
    _, X_test, _, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=args.test_size,
        random_state=42,
        stratify=df["label"],
    )

    pipeline = load_pipeline(args.model_path)
    predictions = pipeline.predict(X_test)

    print(classification_report(y_test, predictions, target_names=["ham", "spam"], zero_division=0))
    print("Confusion matrix [[ham->ham, ham->spam], [spam->ham, spam->spam]]:")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()
