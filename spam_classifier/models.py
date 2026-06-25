from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


def get_model(model_name):
    name = model_name.lower()
    if name in {"lr", "logreg", "logistic_regression"}:
        return LogisticRegression(
            C=10,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
    if name in {"nb", "naive_bayes"}:
        return MultinomialNB(alpha=0.5)
    if name in {"svm", "linear_svm"}:
        return LinearSVC(C=1.0, class_weight="balanced", random_state=42)
    if name in {"knn", "k_nearest_neighbors"}:
        return KNeighborsClassifier(
            n_neighbors=15,
            weights="uniform",
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
    if name in {"rf", "random_forest"}:
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42,
        )

    supported = "lr, nb, svm, knn, rf"
    raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")
