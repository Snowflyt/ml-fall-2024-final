import math
from typing import Literal

import torch
from torch import nn

from ml_fall_2024_final.constants import (
    INPUT_WINDOW,
    OUTPUT_WINDOW_LONG,
    OUTPUT_WINDOW_SHORT,
)
from ml_fall_2024_final.dataset import BikeDataset


class PositionalEncoding(nn.Module):
    def __init__(self, feature_size: int):
        """
        动态正弦余弦位置编码。

        :param feature_size: 输入的特征维度。
        """
        super().__init__()

        # 位置编码的大小由输入的 feature_size 来决定
        self.feature_size = feature_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        为输入张量添加位置编码。

        :param x: 输入张量, 形状为 (batch_size, seq_len, feature_size)。
        :return: 添加了位置编码的张量, 形状为 (batch_size, seq_len, feature_size)。
        """
        _batch_size, seq_len, _ = x.size()

        # 动态生成位置编码
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, self.feature_size, 2).float() * -(math.log(10000.0) / self.feature_size)
        )  # (feature_size / 2)

        # 计算正弦和余弦位置编码
        pe = torch.zeros(seq_len, self.feature_size)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置: cos
        pe = pe.unsqueeze(0)  # (1, seq_len, feature_size)

        # 将位置编码加到输入上
        return x + pe.to(x.device)  # (batch_size, seq_len, feature_size)


class TransformerTimeSeries(nn.Module):
    """
    基于 Transformer 的时间序列预测模型, 输出整个序列。

    :param output_window: 输出序列的长度。
    :param dropout: Dropout 概率。
    """

    def __init__(
        self,
        *,
        output_window: Literal["short", "long"],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_window = INPUT_WINDOW
        self.output_window = OUTPUT_WINDOW_SHORT if output_window == "short" else OUTPUT_WINDOW_LONG

        if output_window == "short":
            d_model = 64
            nhead = 4
            dim_feedforward = 256
            num_layers = 6
        else:
            d_model = 128
            nhead = 8
            dim_feedforward = 1024
            num_layers = 8

        # 规范化层 (LayerNorm) 保证稳定性
        self.norm = nn.LayerNorm(len(BikeDataset.features))
        # 线性层将输入特征转换为 d_model 维度
        self.input_linear = nn.Linear(len(BikeDataset.features), d_model)
        # 位置编码 (Positional Encoding)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer 编码器层
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # 通过线性层生成最终的预测输出
        if output_window == "short":
            self.decoder = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, OUTPUT_WINDOW_SHORT),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(d_model, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, OUTPUT_WINDOW_LONG),
            )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        :param src: 输入张量, 形状为 (batch_size, input_window, feature_size)。
        :return: 预测输出, 形状为 (batch_size, output_window)。
        """

        # 输入先通过正则化, 然后通过线性层调整维度并进行位置编码
        src = self.norm(src)
        src = self.input_linear(src)
        src = self.pos_encoder(src)

        # 通过 Transformer 编码器
        memory = self.transformer_encoder(src)

        # 加入残差连接, Transformer 输出与输入相加
        memory += src  # 残差连接

        # 取 Transformer 输出的最后一个时间步
        last_hidden = memory[:, -1, :]  # (batch_size, feature_size)

        # 通过解码器生成最终的预测
        out = self.decoder(last_hidden)  # (batch_size, output_window)

        return out.view(out.size(0), self.output_window)
