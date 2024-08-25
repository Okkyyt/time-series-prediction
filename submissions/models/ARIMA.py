import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./stock_price.csv')
print(data)
# print(data.columns)

#欠損値なし
# print(data.isnull().sum())

# プロット
# plt.figure(figsize=(10,6))
# plt.plot(data['日付け'],data['終値'])
# plt.plot(data['日付け'],data['始値'])
# plt.plot(data['日付け'],data['安値'])
# plt.plot(data['日付け'],data['高値'])
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
data['曜日'] = pd.to_datetime(data['日付け']).dt.weekday
data['週'] = pd.to_datetime(data['日付け']).dt.isocalendar().week
print(data)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

def plot_data_of_month(years):
    plt.figure(figsize=(18, 12))

    cols = 3 
    rows = len(years) // cols + 1

    for i, year in enumerate(years):
        yearly_data = data_of_month[data_of_month['年'] == year]
        plt.subplot(rows, cols, i + 1)

        plt.plot(yearly_data['月'], yearly_data['終値'], marker='o', label='終値')
        plt.plot(yearly_data['月'], yearly_data['始値'], marker='o', label='始値')
        plt.plot(yearly_data['月'], yearly_data['高値'], marker='o', label='高値')
        plt.plot(yearly_data['月'], yearly_data['安値'], marker='o', label='安値')

        plt.title(year)
        plt.xlabel('月')
        plt.ylabel('価格')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(18, 12))
    for i, year in enumerate(years):
        yearly_data = data_of_month[data_of_month['年'] == year]
        plt.subplot(rows, cols, i + 1)  # サブプロットの位置を設定

        plt.plot(yearly_data['月'], yearly_data['変化率 %'], marker='o', label='変化率')
        plt.title(year)
        plt.xlabel('月')
        plt.ylabel('%')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

years_to_plot = data_of_month[data_of_month['年'].astype(int) >= 2012 ]['年'].unique()
plot_data_of_month(years_to_plot)

#１月よりも１２月のほうが値が高く成長している場合が多い。８月を境に値が急激に変わることが多い。まれに、１月から株価が減少傾向になる場合があるがこれは秋、冬まで続きそこから大幅な上昇がみられると予測する。

# def plot_year_of_date(years):
#     cols = 3  # 列数を指定
#     rows = len(years) // cols + 1
    
#     plt.figure(figsize=(10,6))
#     for i, year in enumerate(years):
#         yearly_data = data[data['年'] == year]
#         plt.subplot(rows, cols, i + 1)

#         plt.plot(yearly_data['日付け'],yearly_data['終値'])
#         plt.plot(yearly_data['日付け'],yearly_data['始値'])
#         plt.plot(yearly_data['日付け'],yearly_data['高値'])
#         plt.plot(yearly_data['日付け'],yearly_data['安値'])

#         plt.title(year)
#         plt.xlabel('日付け')
#         plt.ylabel('価格')
#         plt.grid(True)
    
#     plt.show()

# years_to_plot = data[data['年'].astype(int) >= 2012]['年'].unique()
# plot_year_of_date(years_to_plot)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#季節性の確認
#参考資料　https://qiita.com/satshout/items/1f9c2add8a717d7d8d0b

from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def research_stl(data):
    data['年-月'] = pd.to_datetime(data['年-月'])
    data.set_index('年-月', inplace=True)

    # STL分解を実行
    stl = STL(data['終値'], seasonal=13).fit()

    # 結果をプロット
    fig, ax = plt.subplots(4, 1, figsize=(10, 6), sharex=True)

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

def reserch_acf_pacf(data, lags):
    #ACF(自己相関)とPCAF(偏自己相関)を調べる。
    # コレログラムを描画
    fig, ax = plt.subplots(2, 1, figsize=(10,6))
    plot_acf (data["終値"], lags=lags, ax=ax[0]) # 自己相関係数
    plot_pacf(data["終値"], lags=lags, ax=ax[1]) # 偏自己相関係数
    plt.xlabel('Lag [month]')
    plt.tight_layout()
    plt.show()

research_stl(data_of_month)
reserch_acf_pacf(data_of_month, 40)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#移動平均
def moving_average(span, year):
    moving_data = data[data['年'].astype(int) >= year]

    moving_data['始値_移動平均'] = moving_data['始値'].rolling(window=span, min_periods=1).mean()
    moving_data['終値_移動平均'] = moving_data['終値'].rolling(window=span, min_periods=1).mean()
    moving_data['高値_移動平均'] = moving_data['高値'].rolling(window=span, min_periods=1).mean()
    moving_data['安値_移動平均'] = moving_data['安値'].rolling(window=span, min_periods=1).mean()

    return moving_data

span = 200
year = 2012
moving_data = moving_average(span, year)
moving_data = moving_data[::-1].reset_index(drop=True)
print(moving_data)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#モデル作成
from sklearn.model_selection import train_test_split

def split_data(data):
    train_data, test_data = data[0:int(len(data)*0.7)], data[int(len(data)*0.7):]
    return train_data, test_data

train_data, test_data = split_data(data_of_month[data_of_month['年'].astype(int) >= 2012])

print(train_data,test_data)

#ARIMAモデル
from statsmodels.tsa.arima.model import ARIMA

history = [x for x in train_data['終値']]
model_predictions = []

for time_point in range(len(test_data['終値'])):
    model = ARIMA(history, order=(6,1,0))
    model_fit = model.fit()

    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)

    true_test_value = test_data.iloc[time_point]['終値']
    history.append(true_test_value)

plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_data['終値'], color='Red', label='train')
plt.plot(test_data.index, model_predictions, color='Blue', label='prediction')
plt.title('ARIMA model')
plt.legend()
plt.show()

#評価
forecast_period = 30

model = ARIMA(history, order=(6,1,1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=forecast_period)

plt.figure(figsize=(10, 6))
plt.plot(range(len(history)), history, color='Red', label='original')
plt.plot(range(len(history), len(history) + forecast_period), forecast, color='Blue', label='prediction')
plt.title('ARIMA model prediction')
plt.legend()
plt.show()