import re


def normalize_text(text):
    """Lightweight text normalization shared by train and prediction."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\b\d+\b", " NUMBER ", text)
    text = re.sub(r"[^0-9a-zA-Z\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

