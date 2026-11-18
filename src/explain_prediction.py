
import pandas as pd
import joblib

model = joblib.load("models/finai_model.pkl")

df = pd.read_csv("data/cleaned_data.csv")

print("\nMODEL EXPECTED FEATURES:")
print(model.feature_names_in_)

print("\nCSV AVAILABLE FEATURES:")
print(df.columns.tolist())

print("\nMODEL FEATURE COUNT:", len(model.feature_names_in_))
print("CSV FEATURE COUNT:", len(df.columns))
