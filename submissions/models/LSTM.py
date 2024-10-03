import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    data = pd.read_csv('./stock_price.csv')
    # print(data)
    # print(data.columns)

    #欠損値なし
    # print(data.isnull().sum())

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
    data['曜日'] = pd.to_datetime(data['日付け']).dt.weekday
    data['週'] = pd.to_datetime(data['日付け']).dt.isocalendar().week
    # print(data)

    return data
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#月ごとのデータ
def get_data_of_month(data):
    data_of_month = data.groupby(['年','月'])[['終値', '始値', '高値', '安値', '出来高', '変化率 %']].mean()
    data_of_month = data_of_month.reset_index()
    data_of_month['年-月'] = data_of_month['年'] + '-' + data_of_month['月']
    # print(data_of_month)
    return data_of_month

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#フィルター
def filter_data(data,year):
    return data[data['年'] >= year]

#-----------------------------------------------------------
#季節性の確認

from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def research_stl(data):
    data['年-月'] = pd.to_datetime(data['年-月'])
    data.set_index('年-月', inplace=True)

    stl = STL(data['終値'], seasonal=13).fit()
    ig, ax = plt.subplots(4, 1, figsize=(10, 6), sharex=True)

    data["終値"].plot(ax=ax[0], c='black')
    ax[0].set_title("Original Data")

    stl.trend.plot(ax=ax[1], c='black')
    ax[1].set_title("Trend")

    stl.seasonal.plot(ax=ax[2], c='black')
    ax[2].set_title("Seasonal")

    stl.resid.plot(ax=ax[3], c='black')
    ax[3].set_title("Residual")

    plt.tight_layout()
    plt.show()

def scaling(data):
    data = data['終値'].values
    data = data.reshape(-1,1)
    return data


#-----------------------------------------------------------
#データの実行
data = get_data()
data_of_month = get_data_of_month(data)
data = filter_data(data,'2012')
data_of_month = filter_data(data_of_month,'2012')
print(data_of_month.head())
print(data.head())
research_stl(data_of_month)
