import pandas as pd
df_aa = pd.read_csv('./submit0616aa.csv')
df_bb = pd.read_csv('./submit0616bb.csv')

df = pd.merge(df_aa, df_bb, how='outer', on='file_name')
df = df[df['file_code_x'] != df['file_code_y']]
df.to_csv('./diff.csv', index=None)
print(df.head(10))
# for i, row in df_aa.iterrows():
#
#     print(i, '\n',  row)
#     if i > 100:
#         break
