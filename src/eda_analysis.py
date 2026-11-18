
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/cleaned_data.csv")

# Remove the first row if it contains the original column labels
if str(df.iloc[0, 0]).strip().upper() == "ID":
    df = df.iloc[1:].copy()

# Rename columns
df.columns = [
    "ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "Y"
]

# Convert columns into numerical data
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove ID because it is not a useful prediction feature
df = df.drop(columns=["ID"])

# Check dataset
print("Dataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

print("\nStatistical Summary:")
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(df.corr(), cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Credit Default Correlation Heatmap")
plt.tight_layout()
plt.show()

# Target distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Y")
plt.title("Credit Default Distribution")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.show()

# Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="AGE", bins=20, kde=True)
plt.title("Customer Age Distribution")
plt.show()

print("\nEDA Analysis Completed!")
