import pandas as pd

from .paths import DEFAULT_DATASET_PATH


def load_dataset(path=DEFAULT_DATASET_PATH, drop_duplicates=True, max_text_length=5000):
    """Load and lightly clean the combined spam dataset."""
    df = pd.read_csv(path)
    required_columns = {"text", "label"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.copy()
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    if max_text_length:
        df["text"] = df["text"].str.slice(0, max_text_length)

    if drop_duplicates:
        df = df.drop_duplicates(subset=["text", "label"])

    return df.reset_index(drop=True)


def dataset_summary(df):
    text_lengths = df["text"].astype(str).str.len()
    return {
        "rows": int(len(df)),
        "ham": int((df["label"] == 0).sum()),
        "spam": int((df["label"] == 1).sum()),
        "missing_text": int(df["text"].isna().sum()),
        "duplicate_text": int(df["text"].duplicated().sum()),
        "avg_text_length": float(text_lengths.mean()),
        "max_text_length": int(text_lengths.max()),
    }

