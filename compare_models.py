import argparse
import sys
import time

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from spam_classifier.data import load_dataset
from spam_classifier.paths import DEFAULT_DATASET_PATH, REPORT_DIR
from spam_classifier.pipeline import build_pipeline


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare multiple spam classifier models.")
    parser.add_argument("--data", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--models", nargs="+", default=["lr", "nb", "svm", "knn", "rf"])
    parser.add_argument("--max-features", type=int, default=3000)
    parser.add_argument("--test-size", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=args.test_size,
        random_state=42,
        stratify=df["label"],
    )

    rows = []
    for model_name in args.models:
        start = time.time()
        pipeline = build_pipeline(model_name=model_name, max_features=args.max_features)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        rows.append(
            {
                "model": model_name,
                "vectorizer": "tfidf",
                "max_features": args.max_features,
                "accuracy": accuracy_score(y_test, predictions),
                "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
                "precision_spam": precision_score(y_test, predictions, zero_division=0),
                "recall_spam": recall_score(y_test, predictions, zero_division=0),
                "f1_spam": f1_score(y_test, predictions, zero_division=0),
                "train_seconds": round(time.time() - start, 2),
            }
        )

    results = pd.DataFrame(rows).sort_values("f1_spam", ascending=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "model_comparison.csv"
    results.to_csv(output_path, index=False)
    print(results.to_string(index=False))
    print(f"Saved comparison: {output_path}")


if __name__ == "__main__":
    main()
