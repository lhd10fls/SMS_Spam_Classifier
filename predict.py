import argparse
import sys

from spam_classifier.paths import DEFAULT_MODEL_PATH
from spam_classifier.pipeline import load_pipeline


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict whether a message is spam or ham.")
    parser.add_argument("message", nargs="+")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    return parser.parse_args()


def main():
    args = parse_args()
    message = " ".join(args.message)
    pipeline = load_pipeline(args.model_path)
    prediction = int(pipeline.predict([message])[0])
    label = "Spam" if prediction == 1 else "Ham"

    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        probability = pipeline.predict_proba([message])[0][prediction]
        print(f"{label} - confidence {probability:.2%}")
    else:
        print(label)


if __name__ == "__main__":
    main()
