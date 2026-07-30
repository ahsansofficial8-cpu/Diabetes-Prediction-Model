import pandas as pd
import numpy as np

# Step 1: Load the CSV file
df = pd.read_csv("diabetes_prediction_dataset.csv")

# === Basic Inspection ===
print("Shape:", df.shape)   # rows & columns
print(df.info())           # column names, data types, null counts
print(df.isnull().sum())   # missing values per column
print(df.describe())       # summary stats
print("Duplicate rows:", df.duplicated().sum())

# === Remove Duplicates ===
df = df.drop_duplicates()

# === Remove Invalid Values ===
df = df[df['age'] > 0]   # Age must be positive
df = df[df['bmi'] > 0]   # BMI must be positive

# === Handle Missing Values ===
# Fill numeric columns with median
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
cat_cols = df.select_dtypes(exclude=[np.number]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# === Encode Smoking History Dynamically ===
unique_smoking = sorted(df['smoking_history'].dropna().unique())
smoking_map = {val: i for i, val in enumerate(unique_smoking)}
print("Smoking map:", smoking_map)
df['smoking_history'] = df['smoking_history'].map(smoking_map)

# === Encode Gender Dynamically ===
unique_gender = sorted(df['gender'].dropna().unique())
gender_map = {val: i for i, val in enumerate(unique_gender)}
print("Gender map:", gender_map)
df['gender'] = df['gender'].map(gender_map)

# === Final Check ===
print("Cleaned dataset shape:", df.shape)
print(df.head())
df.to_csv("diabetes_fully_cleaned.csv", index=False)
print("Cleaned dataset saved as diabetes_fully_cleaned.csv")
plt.figure(figsize=(6,4))
sns.histplot(df['age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.savefig("age_histogram.png")

plt.figure(figsize=(6,4))
sns.histplot(df['bmi'], bins=30, kde=True)
plt.title("BMI Distribution")
plt.savefig("bmi_histogram.png")

plt.figure(figsize=(6,4))
sns.histplot(df['HbA1c_level'], bins=30, kde=True)
plt.title("HbA1c Level Distribution")
plt.savefig("hba1c_histogram.png")

plt.figure(figsize=(6,4))
sns.histplot(df['blood_glucose_level'], bins=30, kde=True)
plt.title("Blood Glucose Level Distribution")
plt.savefig("glucose_histogram.png")

plt.figure(figsize=(6,4))
df['smoking_history'].value_counts().plot(kind='bar')
plt.title("Smoking History Distribution")
plt.savefig("smoking_bar.png")

plt.figure(figsize=(6,4))
df['gender'].value_counts().plot(kind='bar')
plt.title("Gender Distribution")
plt.savefig("gender_bar.png")