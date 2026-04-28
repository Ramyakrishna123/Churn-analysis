import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
import os

RAW = "data/raw/telecom_churn.csv"
PROC = "data/processed/processed.csv"
MODEL = "models/rf.joblib"
os.makedirs("data/processed", exist_ok=True); os.makedirs("models", exist_ok=True)

# load + simple clean/encode
df = pd.read_csv(RAW)
if "customerID" in df.columns: df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df.get("TotalCharges", pd.Series()), errors="coerce").fillna(0)
df["Churn"] = df["Churn"].map({"Yes":1,"No":0})
df = pd.get_dummies(df, drop_first=True)
df.to_csv(PROC, index=False)

# train
X = df.drop(columns=["Churn"])
y = df["Churn"]
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)
dump(model, MODEL)

# eval (on same data for minimal pipeline)
preds = model.predict(X)
print("accuracy:", round(accuracy_score(y, preds),4))
