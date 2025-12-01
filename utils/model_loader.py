import joblib
from pathlib import Path

def load_models(model_dir="models"):
    return {
        "cardio": joblib.load(Path(model_dir) / "cardio_model_ZB.pkl"),
        "neuro": joblib.load(Path(model_dir) / "neuro_model_ZB.pkl"),
        "hepato": joblib.load(Path(model_dir) / "hepato_model_ZB.pkl"),
    }
