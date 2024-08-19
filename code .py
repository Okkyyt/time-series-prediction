import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./stock_price.csv')
print(data)
# print(data.columns)

#欠損値なし
# print(data.isnull().sum())

# プロット
# plt.figure(figsize=(10,6))
# plt.plot(data['日付け'],data['終値'])
# plt.show()

data['年'] = data['日付け'].str[:4]
data['月'] = data['日付け'].str[5:7]

def convert_volume(value):
    if 'B' in value:
        return float(value.replace('B', '')) * 1000
    elif 'M' in value:
        return float(value.replace('M', '')) 
    else:
        return float(value)
data['出来高'] = data['出来高'].apply(convert_volume)
data['変化率 %'] = data['変化率 %'].str.replace('%','').astype(float)
print(data)
# ------------------------------------------------------------------------------------------------------------------------------------------------------

data_of_month = data.groupby(['年','月'])[['終値', '始値', '高値', '安値', '出来高', '変化率 %']].mean()
data_of_month = data_of_month.reset_index()
data_of_month['年-月'] = data_of_month['年'] + '-' + data_of_month['月']
print(data_of_month)

#年ごとのプロット
plt.figure(figsize=(10,6))
grouped_data = data_of_month.groupby('年')['終値'].mean()
plt.plot(grouped_data.index, grouped_data)
plt.xticks(rotation=90)
plt.show()

# # 月ごとプロット
# plt.figure(figsize=(10,6))
# plt.plot(data_of_month.index,data_of_month['終値'])
# plt.plot(data_of_month.index,data_of_month['始値'])
# plt.plot(data_of_month.index,data_of_month['高値'])
# plt.plot(data_of_month.index,data_of_month['安値'])
# # plt.plot(data_of_month.index,data_of_month['出来高'])
# plt.show()

plt.figure(figsize=(10,6))
plt.plot(data_of_month[data_of_month['年'].astype(int) >= 2020]['年-月'],data_of_month[data_of_month['年'].astype(int) >= 2020]['終値'])
plt.plot(data_of_month[data_of_month['年'].astype(int) >= 2020]['年-月'],data_of_month[data_of_month['年'].astype(int) >= 2020]['始値'])
plt.plot(data_of_month[data_of_month['年'].astype(int) >= 2020]['年-月'],data_of_month[data_of_month['年'].astype(int) >= 2020]['高値'])
plt.plot(data_of_month[data_of_month['年'].astype(int) >= 2020]['年-月'],data_of_month[data_of_month['年'].astype(int) >= 2020]['安値'])
plt.xticks(rotation=90)
plt.show()










