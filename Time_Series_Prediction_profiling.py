import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os
from torch.profiler import profile, record_function, ProfilerActivity
import time
from scipy.signal import savgol_filter

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
df = pd.read_csv('combined_data.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H.%M')
df.set_index('Date', inplace=True)
df = df.resample('15min').mean()

window_length = 11  # Window length: Must be an odd number (e.g., 7, 11, 15, 21).
polyorder = 3       # Polynomial order: Typically 2 or 3. Must be less than window_length.
# Ensure window_length does not exceed the number of rows in the DataFrame.
if len(df) >= window_length:
    df_filtered = pd.DataFrame(
        savgol_filter(df, window_length=window_length, polyorder=polyorder, axis=0),
        index=df.index,
        columns=df.columns
    )
else:
    print(f"Data has too few rows ({len(df)}) for the specified window_length ({window_length}).")
    print("No smoothing applied. df_filtered is a copy of the original df.")
    df_filtered = df.copy()

df = df_filtered

training_set = df['Load'].values.reshape(-1, 1)
plt.plot(training_set, label='Time Series Dataset')
plt.show()

def sliding_windows(data, seq_length, pred_length=4):
    x, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length:i+seq_length+pred_length].reshape(-1)
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)

# Normalize and create sequences
sc = MinMaxScaler()
training_data = sc.fit_transform(training_set)
seq_length = 4 * 24 * 7
pred_length = 4
x, y = sliding_windows(training_data, seq_length, pred_length)

print("x shape:", x.shape)
print("y shape:", y.shape)

# Convert to tensors
X_tensor = torch.tensor(x, dtype=torch.float32)
Y_tensor = torch.tensor(y, dtype=torch.float32)

# Split data
train_size = int(len(y) * 0.67)
trainX, testX = X_tensor[:train_size], X_tensor[train_size:]
trainY, testY = Y_tensor[:train_size], Y_tensor[train_size:]

# Datasets and loaders
batch_size = 1024
train_dataset = TensorDataset(trainX, trainY)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# LSTM Model
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
num_epochs = 1000
learning_rate = 0.01
input_size = 1
hidden_size = 2
num_layers = 1
num_classes = pred_length

# Model setup
lstm = LSTM(num_classes, input_size, hidden_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

model_path = "trained_lstm_model.pth"
if os.path.exists(model_path):
    lstm.load_state_dict(torch.load(model_path, map_location=device))
    lstm.eval()
    print("Loaded trained model.")
else:
    start_time = time.time()
    log_dir = './log/profiler'
    
    # Profiling setup ctrl+shift+P python: launch tensorboard
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for epoch in range(num_epochs):
            lstm.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                with record_function("model_inference"):
                    outputs = lstm(batch_x)
                    loss = criterion(outputs, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                prof.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.5f}")
        torch.save(lstm.state_dict(), model_path)
        print("Trained model saved.")
        
    end_time = time.time()
    print(f"Total training time: {(end_time - start_time):.2f} seconds")

# Prediction
lstm.eval()
dataX_tensor = torch.tensor(x, dtype=torch.float32).to(device)
with torch.no_grad():
    train_predict = lstm(dataX_tensor).cpu().numpy()
    actual = y

data_predict = sc.inverse_transform(train_predict)
dataY_plot = sc.inverse_transform(actual)
dataY_plot_show = dataY_plot[:, 0]

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

# Sliding window sample plots
start_point = 30000
selected_indices = list(range(start_point, start_point + 10))

dataX_cpu = X_tensor.cpu()
for idx, i in enumerate(selected_indices):
    plt.figure(figsize=(8,3))
    input_window = sc.inverse_transform(dataX_cpu[i].numpy())
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
