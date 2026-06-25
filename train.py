import argparse
import json
import sys
import time

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from spam_classifier.data import dataset_summary, load_dataset
from pathlib import Path

from spam_classifier.paths import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, PIPELINE_MODEL_DIR, REPORT_DIR
from spam_classifier.pipeline import build_pipeline, save_pipeline


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a spam/ham classifier pipeline.")
    parser.add_argument("--model", default="lr", choices=["lr", "nb", "svm", "knn", "rf"])
    parser.add_argument("--data", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--max-features", type=int, default=3000)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def evaluate_split(name, y_true, y_pred):
    return {
        "split": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_spam": precision_score(y_true, y_pred, zero_division=0),
        "recall_spam": recall_score(y_true, y_pred, zero_division=0),
        "f1_spam": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    args = parse_args()
    start = time.time()

    df = load_dataset(args.data)
    summary = dataset_summary(df)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=args.test_size,
        random_state=42,
        stratify=df["label"],
    )

    val_fraction = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction,
        random_state=42,
        stratify=y_train_val,
    )

    pipeline = build_pipeline(model_name=args.model, max_features=args.max_features)
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    val_pred = pipeline.predict(X_val)
    test_pred = pipeline.predict(X_test)

    metrics = [
        evaluate_split("train", y_train, train_pred),
        evaluate_split("validation", y_val, val_pred),
        evaluate_split("test", y_test, test_pred),
    ]

    model_path = (
        PIPELINE_MODEL_DIR / f"{args.model}_tfidf_{args.max_features}_pipeline.joblib"
        if args.output is None
        else Path(args.output)
    )
    save_pipeline(pipeline, model_path)
    if model_path.resolve() != DEFAULT_MODEL_PATH.resolve():
        save_pipeline(pipeline, DEFAULT_MODEL_PATH)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / f"{args.model}_tfidf_{args.max_features}_metrics.csv"
    report_path = REPORT_DIR / f"{args.model}_tfidf_{args.max_features}_report.json"

    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    report = {
        "model": args.model,
        "max_features": args.max_features,
        "model_path": str(model_path),
        "latest_model_path": str(DEFAULT_MODEL_PATH),
        "dataset_summary": summary,
        "metrics": metrics,
        "test_confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "test_classification_report": classification_report(
            y_test,
            test_pred,
            target_names=["ham", "spam"],
            output_dict=True,
            zero_division=0,
        ),
        "training_seconds": round(time.time() - start, 2),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved latest model: {DEFAULT_MODEL_PATH}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved report: {report_path}")
    print(pd.DataFrame(metrics).to_string(index=False))
    print("Test confusion matrix [[ham->ham, ham->spam], [spam->ham, spam->spam]]:")
    print(confusion_matrix(y_test, test_pred))


if __name__ == "__main__":
    main()
