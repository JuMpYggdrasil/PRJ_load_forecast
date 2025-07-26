import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, stacked_size, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.stacked_size = stacked_size
        self.lstm = nn.LSTM(input_size, hidden_size, stacked_size, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.stacked_size, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.stacked_size, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out
    
    
def sliding_windows(data, seq_length, pred_length=4):
    x, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length:i+seq_length+pred_length].reshape(-1)
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)


def time_series_prediction( seq_length = 64,
                            hidden_size = 2,
                            stacked_size = 2,
                            dropout = 0.2):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv('ratch_data.csv')
    # df = pd.read_csv('number_data.csv')
    # df = pd.read_csv('combined_data.csv')
    # df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H.%M')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')
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
    # plt.show()
    plt.close()



    # Normalize and create sequences
    sc = MinMaxScaler()
    training_data = sc.fit_transform(training_set)
    # seq_length = 64 # 4 * 24 * 7
    pred_length = 2 # Predict next n-time steps
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




    # Hyperparameters
    num_epochs = 1000
    learning_rate = 0.001
    input_size = 1
    # hidden_size = 16
    # stacked_size = 2 The number of recurrent layers in the LSTM
    
    num_classes = pred_length

    # Model setup
    lstm = LSTM(num_classes, input_size, hidden_size, stacked_size, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    model_path = "trained_lstm_model.pth"
    if os.path.exists(model_path):
        lstm.load_state_dict(torch.load(model_path, map_location=device))
        lstm.eval()
        print("Loaded trained model.")
    else:
        for epoch in range(num_epochs):
            lstm.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = lstm(batch_x)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.5f}")
        torch.save(lstm.state_dict(), model_path)
        print("Trained model saved.")

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
    # plt.show()
    plt.close()

    # Sliding window sample plots
    start_point = 3412
    selected_indices = list(range(start_point, start_point + 20))

    dataX_cpu = X_tensor.cpu()
    # for idx, i in enumerate(selected_indices):
    #     plt.figure(figsize=(8,3))
    #     input_window = sc.inverse_transform(dataX_cpu[i].numpy())
    #     prediction_window = data_predict[i].reshape(-1, 1)
    #     actual_window = dataY_plot[i].reshape(-1, 1)
    #     plt.plot(range(seq_length), input_window, marker='o', label='Input Window')
    #     plt.plot(range(seq_length, seq_length+pred_length), prediction_window, marker='x', label='Prediction')
    #     plt.plot(range(seq_length, seq_length+pred_length), actual_window, marker='s', label='Actual')
    #     plt.title(f'Sliding Window Sample (Index {i})')
    #     plt.xlabel('Time Step')
    #     plt.ylabel('Load (kW)')
    #     plt.legend()
    #     plt.savefig(f'sliding_window_sample_{i}.png')
    #     # plt.show()
    #     plt.close()


    # Evaluation
    mse = mean_squared_error(dataY_plot[:, 0], data_predict[:, 0])
    mae = mean_absolute_error(dataY_plot[:, 0], data_predict[:, 0])
    r2 = r2_score(dataY_plot[:, 0], data_predict[:, 0])

    print(f"Performance Metrics:")
    print(f"  MSE  = {mse:.4f}")
    print(f"  MAE  = {mae:.4f}")
    print(f"  R²   = {r2:.4f}")

    errors = data_predict[:, 0] - dataY_plot[:, 0]
    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=50, color='skyblue', edgecolor='black')
    plt.title("Prediction Error Distribution")
    plt.xlabel("Error (Predicted - Actual)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig('error_distribution.png')
    # plt.show()
    plt.close()


    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mask = dataY_plot[:, 0] != 0
    mape = mean_absolute_percentage_error(dataY_plot[mask, 0], data_predict[mask, 0])
    accuracy = 100 - mape

    print(f"  MAPE = {mape:.2f}%")
    print(f"  Accuracy (100 - MAPE) = {accuracy:.2f}%")



    results_df = pd.DataFrame({
        'Model': ["LSTM"],          # <--- Make sure this is ["LSTM"]
        'MSE': [mse],               # <--- Make sure this is [mse]
        'MAE': [mae],               # <--- Make sure this is [mae]
        'R2': [r2],                 # <--- Make sure this is [r2]
        'LearningRate': [learning_rate],
        'InputSize': [input_size],
        'HiddenSize': [hidden_size],
        'NumLayers': [stacked_size],
        'SeqLength': [seq_length],
        'Dropout': [dropout],
        'WindowLength': [window_length],
        'PolyOrder': [polyorder],
        'MAPE': [mape],
        'Accuracy': [accuracy]
    })

    # Append to CSV file
    output_file = "model_summary_results.csv"
    if os.path.exists(output_file):
        results_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(output_file, mode='w', header=True, index=False)
        
    return results_df

if __name__ == "__main__":
    seq_lengths = [32, 64, 128]
    hidden_sizes = [2, 4, 16]
    stacked_sizes = [2, 4] # The number of recurrent layers in the LSTM (>1 if use dropout)
    dropouts = [0.1, 0.2, 0.3, 0.4, 0.5] # Dropout rate for LSTM layers
    
    # Loop through all combinations of hyperparameters
    for sl in seq_lengths:
        for hs in hidden_sizes:
            for ss in stacked_sizes:
                for do in dropouts:    
                    print(f"Starting new run with hyperparameters:")
                    print(f"  seq_length: {sl}")
                    print(f"  hidden_size: {hs}")
                    print(f"  stacked_sizes: {ss}")
                    print(f"  dropout: {do}")
                    time_series_prediction(seq_length=sl,
                                            hidden_size=hs,
                                            stacked_size=ss,
                                            dropout=do)
                    
                    file_to_delete = "trained_lstm_model.pth"
                    if os.path.exists(file_to_delete):
                        try:
                            os.remove(file_to_delete)
                        except OSError as e:
                            print(f"Error deleting file '{file_to_delete}': {e}")
