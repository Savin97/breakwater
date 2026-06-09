import pandas as pd

df = pd.read_parquet("output/upcoming_df.parquet")
df = df.sort_values("earnings_date")
df.to_csv("tail.csv",index=False)