import pandas as pd

df = pd.read_csv("price_data.csv")
df['date'] = pd.to_datetime(df['date'])

s1 = df[(df['date'] >= '2021-06-01') & (df['date'] <= '2023-11-01')]['avg_monthly_price'].values
s0 = df[(df['date'] >= '2007-10-01') & (df['date'] <= '2010-08-01')]['avg_monthly_price'].values

print("Length of s1 (2021-06 to 2023-11):", len(s1))
print("Length of s0 (2007-10 to 2010-08):", len(s0))
print("Are they exactly equal?", (s1 == s0).all())
if (s1 == s0).all():
    print("MATCH! The values from 2021-06-01 to 2023-11-01 are an EXACT duplicate of 2007-10-01 to 2010-08-01!")
else:
    print("Difference:", s1 - s0)
