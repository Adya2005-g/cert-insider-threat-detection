import joblib
import pandas as pd

model = joblib.load("model.pkl")

def predict_threat(data):
    df = pd.DataFrame([data])
    result = model.predict(df)
    
    return "Threat" if result[0] == -1 else "Normal"