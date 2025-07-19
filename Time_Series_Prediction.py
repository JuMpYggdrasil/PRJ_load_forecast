import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
df = pd.read_csv('combined_data.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H.%M')
df.set_index('Date', inplace=True)
df = df.resample('15min').mean()

training_set = df['Load'].values.reshape(-1, 1)

plt.plot(training_set, label = 'Time Series Dataset')
plt.show()

def sliding_windows(data, seq_length, pred_length=4):
    x = []
    y = []
    for i in range(len(data) - seq_length - pred_length + 1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length:i+seq_length+pred_length].reshape(-1)
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)

sc = MinMaxScaler()
training_data = sc.fit_transform(training_set)

# Create sliding windows
seq_length = 4*24*7  # 1 weeks of 15-minute intervals
pred_length = 4
x, y = sliding_windows(training_data, seq_length, pred_length)

print("x shape:", x.shape)
print("y shape:", y.shape)
# print("First x sample:\n", x[0])
# print("First y sample:\n", y[0])

train_size = int(len(y) * 0.67)
test_size = len(y) - train_size

# Move tensors to device
dataX = torch.tensor(x, dtype=torch.float32).to(device)
dataY = torch.tensor(y, dtype=torch.float32).to(device)
trainX = torch.tensor(x[0:train_size], dtype=torch.float32).to(device)
trainY = torch.tensor(y[0:train_size], dtype=torch.float32).to(device)
testX = torch.tensor(x[train_size:], dtype=torch.float32).to(device)
testY = torch.tensor(y[train_size:], dtype=torch.float32).to(device)

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out

num_epochs = 1000
learning_rate = 0.01
input_size = 1
hidden_size = 2
num_layers = 1
num_classes = pred_length

lstm = LSTM(num_classes, input_size, hidden_size, num_layers).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Check if trained model exists
model_path = "trained_lstm_model.pth"
if os.path.exists(model_path):
    lstm.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    lstm.eval()
    print("Loaded trained model from trained_lstm_model.pth")
else:
    # Train the model
    for epoch in range(num_epochs):
        outputs = lstm(trainX)
        optimizer.zero_grad()
        loss = criterion(outputs, trainY)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    # Save the trained model to a file
    torch.save(lstm.state_dict(), model_path)
    print("Trained model saved as trained_lstm_model.pth")

    lstm.eval()

# Predict
with torch.no_grad():
    train_predict = lstm(dataX)
    data_predict = train_predict.cpu().numpy()
    dataY_plot = dataY.cpu().numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)
print("dataY_plot shape:", dataY_plot.shape)
dataY_plot_show = dataY_plot[:, 0]
print("Model is on device:", next(lstm.parameters()).device)


# Save and show the main prediction plot
plt.axvline(x=train_size, c='r', linestyle='--')
plt.plot(dataY_plot_show, label='Actual')
plt.plot(data_predict, label='Predicted', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Load (kW)')
plt.legend()
plt.suptitle('Time-Series Prediction')
plt.savefig('time_series_prediction.png')
plt.show()
plt.close()

# Save and show sliding window sample plots from selected indices
start_point = 30000
selected_indices = list(range(start_point, start_point + 10))

for idx, i in enumerate(selected_indices):
    plt.figure(figsize=(8,3))
    input_window = sc.inverse_transform(dataX[i].cpu().numpy())
    prediction_window = data_predict[i].reshape(-1, 1)
    actual_window = dataY_plot[i].reshape(-1, 1)
    plt.plot(range(seq_length), input_window, marker='o', label='Input Window')
    plt.plot(range(seq_length, seq_length+pred_length), prediction_window, marker='x', label='Prediction')
    plt.plot(range(seq_length, seq_length+pred_length), actual_window, marker='s', label='Actual')
    plt.title(f'Sliding Window Sample (Index {i})')
    plt.xlabel('Time Step')
    plt.ylabel('Load (kW)')
    plt.legend()
    plt.savefig(f'sliding_window_sample_{i}.png')
    plt.show()
    plt.close()
