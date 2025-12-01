import numpy as np

def predict_all_toxicities(models, cardio_features, neuro_features, hepato_features):
    cardio = models["cardio"].predict_proba([cardio_features])[0][1]
    neuro  = models["neuro"].predict_proba([neuro_features])[0][1]
    hepato = models["hepato"].predict_proba([hepato_features])[0][1]
    
    return cardio, neuro, hepato
