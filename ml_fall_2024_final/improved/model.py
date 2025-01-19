import torch.nn.functional as F  # noqa: N812
from torch import nn


class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, lstm_hidden_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整维度 (batch, channels, seq_len)
        x = F.relu(self.conv(x))
        x = x.permute(0, 2, 1)  # 调整回原始维度 (batch, seq_len, channels)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])  # 取最后时间步输出
