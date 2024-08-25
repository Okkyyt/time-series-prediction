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

data = get_data()
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#月ごとのデータ
def get_data_of_month(data):
    data_of_month = data.groupby(['年','月'])[['終値', '始値', '高値', '安値', '出来高', '変化率 %']].mean()
    data_of_month = data_of_month.reset_index()
    data_of_month['年-月'] = data_of_month['年'] + '-' + data_of_month['月']
    # print(data_of_month)
    return data_of_month

data_of_month = get_data_of_month(data)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#年ごとのプロット
def plot_overall(data):
    plt.figure(figsize=(10,6))
    grouped_data = data.groupby('年')['終値'].mean()
    plt.plot(grouped_data.index, grouped_data)
    plt.grid()
    plt.xticks(rotation=90)
    plt.show()

plot_overall(data_of_month)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#2012年以降のデータに絞る
def filtered_data(data, year):
    data = data[data['年']>=year]
    return data

data = filtered_data(data,'2012')
data_of_month = filtered_data(data_of_month,'2012')
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

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#データの前処理

#正規化
def scaling(data):
    min_val, max_val = np.min(data["終値"].values), np.max(data["終値"].values)
    scale = max_val - min_val
    data["y"] = (data["終値"].values - min_val) / scale
    print(data)
    return data

data_of_month = scaling(data_of_month)

#trainデータとtestデータで分ける
def train_test_split(data,size):
    train_size = int(len(data) * size)
    train_data, test_data = data[:train_size], data[train_size:]
    return train_data, test_data

train_data, test_data = train_test_split(data_of_month, 0.8)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#SARIMAモデル
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train_data['終値'], 
                order=(1, 1, 1),             # ARIMAの(p,d,q)パラメータ
                seasonal_order=(1, 1, 1, 12), # 季節性の(P,D,Q,m)パラメータ
                enforce_stationarity=False, 
                enforce_invertibility=False)

result = model.fit()

# モデルの要約を表示
print(result.summary())

prediction = result.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, typ='levels')

# プロット
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data['終値'], label='Train')
plt.plot(test_data.index, test_data['終値'], label='Test')
plt.plot(test_data.index, prediction, label='Pred', color='red')
plt.legend()
plt.title('SARIMA Model')
plt.show()

#モデルの評価
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(test_data['終値'], prediction)
mse = mean_squared_error(test_data['終値'], prediction)
rmse = np.sqrt(mse)

print(f"AIC: {result.aic}")
print(f"BIC: {result.bic}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE:{rmse}")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import itertools

#パラメータチューニング
params_1 = {
    'p': [0, 1],
    'd': [0, 1],
    'q': [0, 1],
}
params_2 = {
    'P': [0, 1],
    'D': [0, 1],
    'Q': [0, 1],
    'm': [12],
}

param_combinations = itertools.product(params_1['p'], params_1['d'], params_1['q'], params_2['P'], params_2['D'], params_2['Q'], params_2['m'])

params = []
scores = []

for param in param_combinations:

    model = SARIMAX(train_data['終値'], 
                        order=(param[0], param[1], param[2]), 
                        seasonal_order=(param[3], param[4], param[5], param[6]),
                        enforce_stationarity=False, 
                        enforce_invertibility=False)
    result = model.fit(disp=False)
        
    va_pred = result.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, typ='levels')
    print(va_pred)
    rmse = np.sqrt(mean_squared_error(test_data['終値'], va_pred))
    scores.append((param, rmse))
    params.append(param)

best_params, best_score = scores[np.argmin([score[1] for score in scores])]

best_model = SARIMAX(train_data['終値'], 
                        order=(best_params[0], best_params[1], best_params[2]), 
                        seasonal_order=(best_params[3], best_params[4], best_params[5], best_params[6]),
                        enforce_stationarity=False, 
                        enforce_invertibility=False).fit()

# 最良のパラメータとRMSEを表示
print(f"Best Params: {best_params}")
print(f"RMSE: {best_score}")

# 最良モデルの予測とプロット
prediction = best_model.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, typ='levels')

plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data['終値'], label='Train')
plt.plot(test_data.index, test_data['終値'], label='Test')
plt.plot(test_data.index, prediction, label='Pred', color='red')
plt.legend()
plt.title('SARIMA Model')
plt.show()

# 最良モデルの評価
mae = mean_absolute_error(test_data['終値'], prediction)
mse = mean_squared_error(test_data['終値'], prediction)
rmse = np.sqrt(mse)

print(f"AIC: {best_model.aic}")
print(f"BIC: {best_model.bic}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

#STLを見比べてみる----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
stl_train = STL(test_data['終値'], seasonal=13).fit()
stl_pred = STL(prediction, seasonal=13).fit()

fig, ax = plt.subplots(4, 1, figsize=(10, 6), sharex=True)

test_data["終値"].plot(ax=ax[0], c='blue', label='Original Data')
prediction.plot(ax=ax[0], c='orange' ,label='Prediction')
ax[0].set_title("Data")
ax[0].legend()

stl_train.trend.plot(ax=ax[1], c='blue')   
stl_pred.trend.plot(ax=ax[1], c='orange')
ax[1].set_title("Trend")

stl_train.seasonal.plot(ax=ax[2], c='blue')
stl_pred.seasonal.plot(ax=ax[2], c='orange')
ax[2].set_title("Seasonal")

stl_train.resid.plot(ax=ax[3], c='blue')
stl_pred.resid.plot(ax=ax[3], c='orange')
ax[3].set_title("Residual")

plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#過去3年のデータを細かく調べてみる

# data = get_data()
# data_of_month = get_data_of_month(data)
# data = filtered_data(data ,'2021')
# data_of_month = filtered_data(data_of_month, '2021')
# research_stl(data_of_month)
# reserch_acf_pacf(data_of_month, 20)

# data = data[::-1].reset_index(drop=True) #データの向きの補正

# def moving_average(data, span):
#     plt.figure(figsize=(10, 6))
#     moving_average = data['終値'].rolling(window=span, min_periods=1).mean()
#     plt.plot(data['日付け'], data['終値'], label='Original Data')
#     plt.plot(data['日付け'], moving_average, label=f'Moving Average', color='orange')
#     plt.legend()
#     plt.title(f'Moving Average')
#     plt.show()
#     return moving_average

# moving_average_data = moving_average(data, span=7)
# data['移動平均'] = moving_average_data
# print(data)

# train_data, test_data = train_test_split(data, 0.8)

# # SARIMAモデルの適用 
# model = SARIMAX(train_data['移動平均'], 
#                 order=(6, 1, 0),             
#                 seasonal_order=(1, 1, 1, 12), 
#                 enforce_stationarity=False, 
#                 enforce_invertibility=False)

# result = model.fit()

# print(result.summary())

# prediction = result.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, typ='levels')

# # プロット
# plt.figure(figsize=(10, 6))
# plt.plot(train_data.index, train_data['移動平均'], label='Train')
# plt.plot(test_data.index, test_data['終値'], label='Test')
# plt.plot(test_data.index, prediction, label='Pred', color='red')
# plt.legend()
# plt.title('SARIMA Model')
# plt.show()

# #モデルの評価
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# mae = mean_absolute_error(test_data['終値'], prediction)
# mse = mean_squared_error(test_data['終値'], prediction)
# rmse = np.sqrt(mse)

# print(f"AIC: {result.aic}")
# print(f"BIC: {result.bic}")
# print(f"MAE: {mae}")
# print(f"MSE: {mse}")
# print(f"RMSE:{rmse}")