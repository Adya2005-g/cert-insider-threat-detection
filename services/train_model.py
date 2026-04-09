import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

data = {
    "login_time": [1,2,3,10,12],
    "file_access": [5,3,2,50,60],
    "email_activity": [2,1,1,20,25]
}

df = pd.DataFrame(data)

model = IsolationForest(contamination=0.2)
model.fit(df)

joblib.dump(model, "model.pkl")

print("Model trained successfully!")