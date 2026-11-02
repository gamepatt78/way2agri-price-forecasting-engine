import pandas as pd
df = pd.read_csv('price_data.csv')
for i in range(190, 235):
    for j in range(25, 75):
        if df.iloc[i]['avg_monthly_price'] == df.iloc[j]['avg_monthly_price']:
            k = 0
            while i+k < len(df) and j+k < len(df) and df.iloc[i+k]['avg_monthly_price'] == df.iloc[j+k]['avg_monthly_price']:
                k += 1
            if k > 5:
                print(f"Match of length {k}: i={i} ({df.iloc[i]['date']}) and j={j} ({df.iloc[j]['date']})")
                break
