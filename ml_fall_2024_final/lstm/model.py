from itertools import starmap

import torch
from torch import nn


class LSTMModelWithEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dims, hidden_dim, num_layers, output_dim):
        super().__init__()

        # 创建嵌入层
        self.embeddings = nn.ModuleList(list(starmap(nn.Embedding, embedding_dims)))

        # LSTM 层
        total_embedding_dim = sum(embedding_dim for _, embedding_dim in embedding_dims)
        self.lstm = nn.LSTM(
            input_dim + total_embedding_dim, hidden_dim, num_layers, batch_first=True
        )

        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_cont, x_cat):
        # 嵌入层处理离散型变量
        embedded = [embedding(x_cat[:, :, i]) for i, embedding in enumerate(self.embeddings)]
        embedded = torch.cat(embedded, dim=-1)  # 拼接所有嵌入层输出

        # 拼接连续型特征和嵌入特征
        x = torch.cat((x_cont, embedded), dim=-1)

        # 输入 LSTM
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后时间步的输出

        # 输出层
        return self.fc(out)
