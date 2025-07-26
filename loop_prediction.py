import os
import time
from Time_Series_Prediction import time_series_prediction

if __name__ == "__main__":
    seq_lengths = [4, 32, 64, 128]
    hidden_sizes = [2, 4, 16]
    stacked_sizes = [1, 2, 4] # The number of recurrent layers in the LSTM (>1 if use dropout)
    dropouts = [0, 0.2] # Dropout rate for LSTM layers
    
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