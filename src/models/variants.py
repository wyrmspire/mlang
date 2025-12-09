
import torch
import torch.nn as nn

class CNN_Classic(nn.Module):
    """
    Original Architecture: 20-bar lookback, 4 channels (O,H,L,C).
    2 Conv layers + FC.
    """
    def __init__(self, input_dim=4, seq_len=20):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2), # 20 -> 10
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2), # 10 -> 5
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 5, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (Batch, Seq, Dim) -> (Batch, Dim, Seq)
        x = x.transpose(1, 2)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CNN_Wide(nn.Module):
    """
    Wide Context Architecture: 60-bar lookback.
    Deeper Conv layers.
    """
    def __init__(self, input_dim=4, seq_len=60):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2), # 60 -> 30
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2), # 30 -> 15
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(3), # 15 -> 5
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class LSTM_Seq(nn.Module):
    """
    Recurrent Architecture: 20-bar lookback.
    LSTM layer to capture temporal order.
    """
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (Batch, Seq, Dim)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        last_step = lstm_out[:, -1, :]
        return self.fc(last_step)

class Feature_MLP(nn.Module):
    """
    Feature-based Feed Forward.
    Input: Pre-calculated Technical Indicators (Simulated Dim=10 for now).
    """
    def __init__(self, input_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (Batch, Dim) - No sequence dim
        return self.net(x)

class CNN_Predictive(nn.Module):
    """
    Predictive Model: 20-bar lookback.
    Input-Dim: 5 (Open, High, Low, Close, ATR).
    Predicts probability of 15m Rejection.
    """
    def __init__(self, input_dim=5, seq_len=20): 
        super(CNN_Predictive, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * (seq_len // 2), 64)
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, Time, Feature) -> (Batch, Feature, Time) needed for Conv1d
        # Or (Batch, Feature, Time) already? 
        # Check training script: x = x.permute(0, 2, 1) 
        # In this file, other models assume x is (Batch, Seq, Dim), then transpose.
        # Let's standardize: assume input is (Batch, Seq, Dim)
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dout(x)
        x = self.fc2(x)
        return self.sigmoid(x)
