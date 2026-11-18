
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_excel("data/data.xls")

# 1. Display dataset information
print("Dataset Shape:", df.shape)
print("\nDataset Information:")
df.info()

# 2. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Check duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# 4. Remove duplicate records
df = df.drop_duplicates()

# 5. Handle missing values
for column in df.select_dtypes(include=np.number).columns:
    df[column] = df[column].fillna(df[column].median())

for column in df.select_dtypes(include="object").columns:
    if df[column].isnull().any():
        df[column] = df[column].fillna(df[column].mode()[0])

# 6. Check cleaned dataset
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCleaned Dataset Shape:", df.shape)

# 7. Save cleaned dataset
df.to_csv("data/cleaned_data.csv", index=False)

print("\nCleaned dataset saved successfully!")
