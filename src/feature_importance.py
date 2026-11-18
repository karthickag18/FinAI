
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("models/finai_model.pkl")

# Get feature importance
importance = model.feature_importances_

# Get feature names
features = model.feature_names_in_

# Create DataFrame
df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# Sort features
df = df.sort_values(
    by="Importance",
    ascending=False
)

print(df.head(10))

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(df["Feature"].head(10),
         df["Importance"].head(10))

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 10 Important Features")

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
