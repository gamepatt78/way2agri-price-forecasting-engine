from pathlib import Path

import numpy as np
import pandas as pd


data_path = Path(__file__).resolve().parents[1] / "data" / "price_data.csv"
df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

suspect = df.loc[
    df["date"].between("2021-06-01", "2023-11-01"),
    "avg_monthly_price",
].to_numpy()
historical = df.loc[
    df["date"].between("2007-10-01", "2010-03-01"),
    "avg_monthly_price",
].to_numpy()

if len(suspect) != 30 or len(historical) != 30:
    raise ValueError("Expected two 30-month windows; verify the dataset date range and monthly frequency.")

mismatch_count = int(np.count_nonzero(suspect != historical))
prices = df["avg_monthly_price"].to_numpy()
matching_windows = []
for start in range(len(prices) - len(suspect) + 1):
    end = start + len(suspect)
    if start < 227 and end > 197:
        continue
    if np.array_equal(prices[start:end], suspect):
        matching_windows.append((df.loc[start, "date"], df.loc[end - 1, "date"]))

print(f"Compared 2021-06 through 2023-11 with 2007-10 through 2010-03: {mismatch_count} of 30 values differ.")
print(f"Exact matching 30-month windows elsewhere in the supplied series: {len(matching_windows)}")
