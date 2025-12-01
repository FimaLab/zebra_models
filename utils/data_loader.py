import pandas as pd

def read_profile_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)
    
    if "Drug" not in df.columns:
        raise ValueError("В файле должна быть колонка 'Drug'")

    return df
