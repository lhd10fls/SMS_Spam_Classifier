import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline

from .models import get_model
from .preprocessing import normalize_text


def build_pipeline(model_name="lr", embed_type="tfidf", max_features=3000, ngram_range=(1, 2)):
    if embed_type == "tfidf":
        vectorizer = TfidfVectorizer(
            preprocessor=normalize_text,
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
        )
    else:
        vectorizer = CountVectorizer(
            preprocessor=normalize_text,
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
        )
        
    return Pipeline(
        steps=[
            ("embed", vectorizer),
            ("model", get_model(model_name)),
        ]
    )


def save_pipeline(pipeline, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path):
    return joblib.load(path)

