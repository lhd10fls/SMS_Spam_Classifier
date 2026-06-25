from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
MODEL_DIR = PROJECT_ROOT / "Model"
PIPELINE_MODEL_DIR = MODEL_DIR / "Pipeline"
REPORT_DIR = PROJECT_ROOT / "reports"

DEFAULT_DATASET_PATH = DATA_DIR / "combined_spam_dataset.csv"
DEFAULT_MODEL_PATH = PIPELINE_MODEL_DIR / "spam_pipeline.joblib"
DEFAULT_METRICS_PATH = REPORT_DIR / "metrics.csv"

