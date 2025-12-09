import torch
import torch.nn as nn

class TradeCNN(nn.Module):
    def __init__(self, input_len=20, input_channels=4):
        super(TradeCNN, self).__init__()
        # Input shape: (Batch, Channels, Length) -> (B, 4, 20)
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2) # Output len -> 10
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Pool again? Output len -> 5
        
        self.fc1 = nn.Linear(64 * 5, 32) 
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape should vary but we expect (Batch, 4, 20) in training loop
        # Check if permute needed if input is (Batch, 20, 4)
        if x.shape[1] != 4:
            x = x.permute(0, 2, 1) 
            
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
