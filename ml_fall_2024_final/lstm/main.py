from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ml_fall_2024_final.constants import OUTPUT_WINDOW_LONG, OUTPUT_WINDOW_SHORT
from ml_fall_2024_final.dataset import read_test_data, read_train_data
from ml_fall_2024_final.lstm.model import LSTMModelWithEmbedding
from ml_fall_2024_final.lstm.preprocess import preprocess_data
from ml_fall_2024_final.lstm.trainer import train_model
from ml_fall_2024_final.utils import save_fig, save_model

OUTPUT_WINDOW: Literal["short", "long"] = "long"
OUTPUT_WINDOW_LENGTH = OUTPUT_WINDOW_SHORT if OUTPUT_WINDOW == "short" else OUTPUT_WINDOW_LONG


def start():
    """Launched with `poetry run lstm` at root level"""

    # read csv
    data = read_train_data().dropna().drop(columns=["dteday"])

    # 指定离散型变量列
    discrete_cols = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]

    X_cont_train, X_cat_train, y_train = preprocess_data(
        data=data,
        target_col="cnt",
        discrete_cols=discrete_cols,
        past_steps=OUTPUT_WINDOW_LENGTH,
        future_steps=OUTPUT_WINDOW_LENGTH,
    )

    test_data = read_test_data().dropna().drop(columns=["dteday"])

    X_cont_test, X_cat_test, y_test = preprocess_data(
        data=test_data,
        target_col="cnt",
        discrete_cols=discrete_cols,
        past_steps=OUTPUT_WINDOW_LENGTH,
        future_steps=OUTPUT_WINDOW_LENGTH,
    )

    # 转换为 Tensor
    X_cont_train_tensor = torch.tensor(X_cont_train, dtype=torch.float32)
    X_cat_train_tensor = torch.tensor(X_cat_train, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_cont_test_tensor = torch.tensor(X_cont_test, dtype=torch.float32)
    X_cat_test_tensor = torch.tensor(X_cat_test, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # 创建 DataLoader
    train_dataset = TensorDataset(X_cont_train_tensor, X_cat_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_cont_test_tensor, X_cat_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset)

    # 初始化模型
    input_dim = X_cont_train.shape[2]  # 连续特征数
    hidden_dim = 64
    num_layers = 3
    output_dim = OUTPUT_WINDOW_LENGTH
    embedding_dims = [(4, 2), (3, 2)]  # 示例: 嵌入尺寸 (类别数量, 嵌入维度)

    model = LSTMModelWithEmbedding(input_dim, embedding_dims, hidden_dim, num_layers, output_dim)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, epochs=20)
    save_model(model, f"lstm_{OUTPUT_WINDOW}")

    # 测试模型
    model.eval()
    test_loss = 0.0
    predicts = []
    real = []
    mae_list = []
    mse_list = []
    with torch.no_grad():
        for X_cont_batch, X_cat_batch, y_batch in test_loader:
            outputs = model(X_cont_batch, X_cat_batch)
            predicts.append(outputs)
            real.append(y_batch)
            loss = criterion(outputs, y_batch)
            for i in range(len(y_batch)):
                mae_list.append(criterion_mae(outputs[i], y_batch[i]).item())
                mse_list.append(criterion(outputs[i], y_batch[i]).item())
            test_loss += loss.item()
    print(f"Test Loss: {test_loss / len(test_loader)}")

    # 打印 MAE 和 MSE 的均值和标准差
    selected_mae = []
    selected_mse = []
    for _ in range(5):
        idx = np.random.randint(0, len(predicts))
        selected_mae.append(mae_list[idx])
        selected_mse.append(mse_list[idx])
    mae_std = np.std(selected_mae)
    mse_std = np.std(selected_mse)
    mae_mean = np.mean(selected_mae)
    mse_mean = np.mean(selected_mse)
    print(f"MAE Mean: {mae_mean}, MAE Std: {mae_std}")
    print(f"MSE Mean: {mse_mean}, MSE Std: {mse_std}")

    # 绘制预测结果
    data_x = list(range(OUTPUT_WINDOW_LENGTH))
    data_y = outputs[-1].numpy()
    plt.plot(data_x, data_y, label="LSTM")
    data_y = y_batch[-1].numpy()
    plt.plot(data_x, data_y, label="Ground Truth")
    # left up corner
    plt.legend(loc="upper left")
    # background color white
    plt.gca().set_facecolor("white")
    save_fig(plt.gcf(), f"lstm_{OUTPUT_WINDOW}")
