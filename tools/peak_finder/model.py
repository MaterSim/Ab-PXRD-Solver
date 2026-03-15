import torch.nn as nn

class PeakFinderCNN(nn.Module):
    def __init__(self, window_size=51):
        
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # window_size -> pool -> pool -> pool
        final_size = window_size // 8 * 32
        
        self.fc1 = nn.Linear(final_size, 64)
        self.fc2 = nn.Linear(64, 2)  # 二分类
    
    def forward(self, x):
        # x: (batch, window_size)
        x = x.unsqueeze(1)  # (batch, 1, window_size)
        
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.flatten(1)
        x = self.dropout(x)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.fc2(x)  
        
        return x  