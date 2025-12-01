from pathlib import Path
import joblib

MODELS_DIR = Path("models")

def load_feature_sets():
    neuro = joblib.load(MODELS_DIR / "neuro_model_ZB.pkl").feature_names_in_
    hepato = joblib.load(MODELS_DIR / "hepato_model_ZB.pkl").feature_names_in_
    cardio = joblib.load(MODELS_DIR / "cardio_model_ZB.pkl").feature_names_in_
    
    return {
        "neuro":  list(neuro),
        "hepato": list(hepato),
        "cardio": list(cardio),
    }
