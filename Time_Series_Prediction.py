import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import os




# Load data
df = pd.read_csv('combined_data.csv')
# df = pd.read_csv('number_data.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H.%M')
df.set_index('Date', inplace=True)
df = df.resample('15min').mean()

training_set = df['Load'].values.reshape(-1, 1)

# training_set = pd.read_csv('airline-passengers.csv')
# training_set = training_set.iloc[:,1:2].values

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
seq_length = 24  # the number of previous time steps used as input features to predict the next value.
pred_length = 4   # number of steps to predict
x, y = sliding_windows(training_data, seq_length, pred_length)

print("x shape:", x.shape)  # (num_samples, seq_length, 1)
print("y shape:", y.shape)  # (num_samples, pred_length)
print("First x sample:\n", x[0])
print("First y sample:\n", y[0])

train_size = int(len(y) * 0.67)
test_size = len(y) - train_size

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

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
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out
    
num_epochs = 500
learning_rate = 0.01

input_size = 1
hidden_size = 2
num_layers = 1

num_classes = pred_length  # output 4 values

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Check if trained model exists
model_path = "trained_lstm_model.pth"
if os.path.exists(model_path):
    lstm.load_state_dict(torch.load(model_path))
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
    
train_predict = lstm(dataX)

data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

# Save and show the main prediction plot
plt.axvline(x=train_size, c='r', linestyle='--')
plt.plot(dataY_plot, label='Actual')
plt.plot(data_predict, label='Predicted')
plt.xlabel('Time Step')
plt.ylabel('Load (kW)')
plt.legend()
plt.suptitle('Time-Series Prediction')
plt.savefig('time_series_prediction.png')  # Save the main plot
plt.show()
plt.close()

# Save and show sliding window sample plots from selected indices
start_point = 30000
selected_indices = list(range(start_point, start_point + 12))

for idx, i in enumerate(selected_indices):
    plt.figure(figsize=(8,3))
    # Get input window from dataX
    input_window = sc.inverse_transform(dataX[i].cpu().numpy())
    # Get model prediction and ground truth for this window
    prediction_window = data_predict[i].reshape(-1, 1)
    actual_window = dataY_plot[i].reshape(-1, 1)
    # Plot input window
    plt.plot(range(seq_length), input_window, marker='o', label='Input Window')
    # Plot prediction window
    plt.plot(range(seq_length, seq_length+pred_length), prediction_window, marker='x', label='Prediction')
    # Plot actual future values
    plt.plot(range(seq_length, seq_length+pred_length), actual_window, marker='s', label='Actual')
    plt.title(f'Sliding Window Sample (Index {i})')
    plt.xlabel('Time Step')
    plt.ylabel('Load (kW)')
    plt.legend()
    plt.savefig(f'sliding_window_sample_{i}.png')
    plt.show()
    plt.close()
