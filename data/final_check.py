import os
import pandas as pd
import csv
file_dir = 'final.xlsx'
# read xlsx
df = pd.read_excel(file_dir)
# pd to dict
#df = df.to_dict()
#print(df)
header = list(df.columns)
for i, head in enumerate(header) :
    if i%3 == 0 :
        block_name = head.split('_image')[0]
    else :
        header[i] = block_name + '_' + head
df.columns = header
print(df)
# save to csv
df.to_csv('final.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)