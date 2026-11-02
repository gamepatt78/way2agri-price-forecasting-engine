import pandas as pd
import numpy as np

df = pd.read_csv("price_data.csv")
print("Data Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
print("\nInfo:")
df['date'] = pd.to_datetime(df['date'])
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nDescriptive Statistics:\n", df['avg_monthly_price'].describe())

# Check date frequency / missing dates
df = df.sort_values('date').reset_index(drop=True)
date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='MS')
print("\nExpected months:", len(date_range), "Actual months:", len(df))
missing_dates = set(date_range) - set(df['date'])
print("Missing dates in sequence:", missing_dates)

# Check duplicates
print("Duplicate dates:", df['date'].duplicated().sum())

# Sudden jumps or anomalies:
df['pct_change'] = df['avg_monthly_price'].pct_change()
df['diff'] = df['avg_monthly_price'].diff()
print("\nTop 5 absolute monthly price changes:")
print(df.reindex(df['diff'].abs().sort_values(ascending=False).index).head(10)[['date', 'avg_monthly_price', 'diff', 'pct_change']])
