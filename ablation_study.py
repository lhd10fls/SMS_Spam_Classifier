import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score

from spam_classifier.data import load_dataset
from spam_classifier.pipeline import build_pipeline
from spam_classifier.paths import REPORT_DIR

def evaluate_model(pipeline, X, y):
    y_pred = pipeline.predict(X)
    f1 = f1_score(y, y_pred, average="weighted")
    bal_acc = balanced_accuracy_score(y, y_pred)
    return f1, bal_acc

def run_ablation(model_name):
    print(f"\n{'='*40}")
    print(f"Running Ablation Study for: {model_name.upper()}")
    print(f"{'='*40}")
    
    df = load_dataset()
    X_train, X_temp, y_train, y_temp = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    results = []
    
    for embed_type in ["tfidf", "bow"]:
        for max_f in [1000, 3000]:
            config_name = f"{model_name.upper()}-{embed_type.upper()}{max_f}"
            print(f"Training {config_name}...", end="", flush=True)
            
            start_time = time.time()
            pipeline = build_pipeline(model_name=model_name, embed_type=embed_type, max_features=max_f)
            
            # FIT
            pipeline.fit(X_train, y_train)
            
            # EVALUATE
            train_f1, train_bal_acc = evaluate_model(pipeline, X_train, y_train)
            val_f1, val_bal_acc = evaluate_model(pipeline, X_val, y_val)
            test_f1, test_bal_acc = evaluate_model(pipeline, X_test, y_test)
            
            elapsed = time.time() - start_time
            print(f" done in {elapsed:.1f}s")
            
            results.append({
                "Embed": config_name,
                "Train F1": round(train_f1, 4),
                "Train bal acc": round(train_bal_acc, 4),
                "Val F1": round(val_f1, 4),
                "Val bal acc": round(val_bal_acc, 4),
                "Test F1": round(test_f1, 4),
                "Test bal acc": round(test_bal_acc, 4)
            })
            
    df_results = pd.DataFrame(results)
    
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"{model_name}_ablation_table.csv"
    df_results.to_csv(out_path, index=False)
    
    print(f"\nSaved results to {out_path}")
    print(df_results.to_string(index=False))
    return df_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="knn", choices=["lr", "nb", "svm", "knn", "rf", "all"])
    args = parser.parse_args()
    
    models = ["lr", "nb", "svm", "knn", "rf"] if args.model == "all" else [args.model]
    
    for m in models:
        run_ablation(m)
