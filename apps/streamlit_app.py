import json
from pathlib import Path

import streamlit as st

import _bootstrap  # noqa: F401
from spam_classifier.paths import PIPELINE_MODEL_DIR, REPORT_DIR
from spam_classifier.pipeline import load_pipeline


MODEL_OPTIONS = {
    "Logistic Regression": "lr",
    "Naive Bayes": "nb",
    "SVM": "svm",
    "k-NN": "knn",
    "Random Forest": "rf",
}

EXAMPLES = {
    "Spam khuyến mãi": "Win a free iPhone now! Click this link to claim your prize",
    "Tin nhắn bình thường": "Let's meet tomorrow at 8 for coffee",
    "Spam khẩn cấp": "URGENT! Your account has been selected for a cash reward. Reply YES now",
}


def model_path_for(model_key):
    candidates = sorted(
        PIPELINE_MODEL_DIR.glob(f"{model_key}_tfidf_*_pipeline.joblib"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def report_path_for(model_path):
    if not model_path:
        return None
    report_name = model_path.name.replace("_pipeline.joblib", "_report.json")
    return REPORT_DIR / report_name


def load_report(model_path):
    path = report_path_for(model_path)
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


@st.cache_resource
def cached_pipeline(path):
    return load_pipeline(Path(path))


def predict_with_pipeline(pipeline, message):
    prediction = int(pipeline.predict([message])[0])
    label = "Spam" if prediction == 1 else "Ham"
    model = pipeline.named_steps["model"]

    score = None
    score_name = None
    if hasattr(pipeline, "predict_proba") and hasattr(model, "predict_proba"):
        score = float(pipeline.predict_proba([message])[0][1])
        score_name = "Xác suất spam"
    elif hasattr(model, "decision_function"):
        transformed = pipeline.named_steps["tfidf"].transform([message])
        score = float(model.decision_function(transformed)[0])
        score_name = "Decision score"

    return label, score, score_name


def metric_row(report):
    if not report:
        return None
    test_metric = next((item for item in report.get("metrics", []) if item.get("split") == "test"), None)
    return test_metric


def render_metrics(report):
    test_metric = metric_row(report)
    if not test_metric:
        st.caption("Chưa có report metric cho model này.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{test_metric['accuracy']:.2%}")
    c2.metric("Balanced Acc", f"{test_metric['balanced_accuracy']:.2%}")
    c3.metric("F1 Spam", f"{test_metric['f1_spam']:.2%}")


def available_models():
    rows = []
    for display_name, key in MODEL_OPTIONS.items():
        path = model_path_for(key)
        rows.append(
            {
                "display_name": display_name,
                "key": key,
                "path": path,
                "available": path is not None,
            }
        )
    return rows


st.set_page_config(page_title="SMS Spam Classifier", layout="centered")
st.title("SMS Spam Classifier")
st.caption("Chọn thuật toán, nhập nội dung SMS/email và kiểm tra dự đoán spam/ham.")

models = available_models()
available = [item for item in models if item["available"]]

if not available:
    st.error("Chưa có model pipeline trong Model/Pipeline. Hãy train trước bằng train.py.")
    st.code('python train.py --model lr --max-features 3000')
    st.stop()

category = st.selectbox("Danh mục", ["Combined"])
algorithm = st.selectbox(
    "Thuật toán",
    [item["display_name"] for item in available],
)

selected = next(item for item in available if item["display_name"] == algorithm)
selected_path = selected["path"]
report = load_report(selected_path)

example_name = st.selectbox("Ví dụ nhanh", ["Tự nhập"] + list(EXAMPLES))
default_text = "" if example_name == "Tự nhập" else EXAMPLES[example_name]
message = st.text_area(
    "Nội dung",
    value=default_text,
    height=180,
    placeholder="Nhập SMS hoặc email cần kiểm tra...",
)

left, right = st.columns([1, 1])
show_all = right.checkbox("So sánh tất cả model hiện có", value=False)

if left.button("Phân loại", type="primary", disabled=not message.strip()):
    if show_all:
        st.subheader("Kết quả theo từng model")
        for item in available:
            pipeline = cached_pipeline(str(item["path"]))
            label, score, score_name = predict_with_pipeline(pipeline, message)
            with st.container(border=True):
                cols = st.columns([1.2, 1, 1])
                cols[0].metric(item["display_name"], label)
                if score is not None:
                    value = f"{score:.2%}" if score_name == "Xác suất spam" else f"{score:.4f}"
                    cols[1].metric(score_name, value)
                cols[2].caption(f"Artifact: {item['path'].name}")
    else:
        pipeline = cached_pipeline(str(selected_path))
        label, score, score_name = predict_with_pipeline(pipeline, message)
        st.metric("Kết quả", label)
        st.caption(f"Danh mục: {category}")
        st.caption(f"Model sử dụng: {selected_path.name}")
        if score is not None:
            value = f"{score:.2%}" if score_name == "Xác suất spam" else f"{score:.4f}"
            st.caption(f"{score_name}: {value}")

st.divider()
st.subheader("Metric của model đang chọn")
render_metrics(report)

with st.expander("Model khả dụng"):
    for item in models:
        status = "Có sẵn" if item["available"] else "Chưa train"
        path = item["path"].name if item["path"] else "N/A"
        st.write(f"- {item['display_name']}: {status} - `{path}`")
