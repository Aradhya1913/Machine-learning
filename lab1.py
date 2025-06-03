import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv("housing.csv")
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

# Data overview
print(f"Dataset Shape: {df.shape}\n")
print("Basic Statistics:\n", df.describe().T)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

Numerical = df.select_dtypes(include=[np.number]).columns
print(Numerical)

for col in Numerical:
    plt.figure(figsize=(10,6))
    df[col].plot(kind= 'hist', title =col, bins = 60, edgecolor='black')
    plt.show()
for col in Numerical:
    plt.figure(figsize=(6,6))
    sns.boxplot(df[col],color='red')
    plt.title(col)
    plt.ylabel(col)
    plt.show()