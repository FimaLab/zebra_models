import numpy as np

def proba_to_score(proba):
    return int(np.clip(round(proba * 10), 0, 10))

def toxicity_label(score):
    if score < 6:
        return "Нетоксичен"
    elif score < 8:
        return "Слаботоксичен"
    else:
        return "Токсичен"
