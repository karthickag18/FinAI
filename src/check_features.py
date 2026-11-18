
import pandas as pd
import joblib

model = joblib.load("models/finai_model.pkl")
df = pd.read_csv("data/cleaned_data.csv")

print("MODEL FEATURES:")
print(list(model.feature_names_in_))

print("\nDATASET COLUMNS:")
print(list(df.columns))

print("\nMODEL SHAPE:", len(model.feature_names_in_))
print("DATASET SHAPE:", len(df.columns))
