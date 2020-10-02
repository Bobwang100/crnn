import pandas as pd
df = pd.read_csv('./submit0615.csv')
df = df.sort_values('file_name')
df.to_csv('./submit_0616a.csv', index=None)
print(df.head(10))