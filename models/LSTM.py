#LSTM
import torch
from torch import nn
from torch import optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------------------------------------------------------
#データの読み込み
data = get_data()
data_of_month = get_data_of_month(data)
data = filtered_data(data ,'2021')
data_of_month = filtered_data(data_of_month, '2021')
data = data[::-1].reset_index(drop=True) #データの向きの補正
data['移動平均'] = moving_average(data, span=7)
data['diff_前日差'] = data['終値'].diff().fillna(0)
data['終値-移動平均'] = data['終値'] - data['移動平均']
print(data)#確認

#------------------------------------------------------------------------------------------------------------------------------------------
#LSTM
train_data, test_data = train_test_split(data, 0.8)
scaler = MinMaxScaler(feature_range=(0, 1))

# 特徴量として使用するカラムを指定
features = ['始値', '高値', '安値', '出来高', '月', '曜日', '週', '移動平均', 'diff_前日差', '終値-移動平均']
target = '終値'

# 特徴量とターゲットをスケーリング
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data[features + [target]])
test_data_scaled = scaler.transform(test_data[features + [target]])

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length, :-1]  # 特徴量のシーケンス
        target = data[i+seq_length, -1]  # 対応するターゲット（終値）
        sequences.append(seq)
        targets.append(target)
    return torch.FloatTensor(sequences), torch.FloatTensor(targets)

seq_length = 30  # 例: 30日分のシーケンス
X_train, y_train = create_sequences(train_data_scaled, seq_length)
X_test, y_test = create_sequences(test_data_scaled, seq_length)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size)

        lstm_out, _ = self.lstm(x, (h_0, c_0))
        out = self.linear(lstm_out[:, -1, :])  # 最後のタイムステップのみを使う
        return out

input_size = len(features)
hidden_layer_size = 50
output_size = 1

model = LSTMModel(input_size, hidden_layer_size, output_size)
# 損失関数と最適化関数の定義
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
model.eval()

with torch.no_grad():
    y_test_pred = model(X_test)
    test_loss = criterion(y_test_pred, y_test)
    print(f'Test Loss: {test_loss.item()}')

    # 予測値の逆正規化
    y_test_pred = y_test_pred.numpy()
    y_test_pred = scaler.inverse_transform(np.concatenate((X_test[:, -1, :], y_test_pred.reshape(-1, 1)), axis=1))[:, -1]

    y_test = y_test.numpy()
    y_test = scaler.inverse_transform(np.concatenate((X_test[:, -1, :], y_test.reshape(-1, 1)), axis=1))[:, -1]

    plt.plot(y_test, label='Actual')
    plt.plot(y_test_pred, label='Predicted')
    plt.legend()
    plt.show()
