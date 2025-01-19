from typing import cast

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from ml_fall_2024_final.type import WindowsData
from ml_fall_2024_final.utils import set_seed


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
) -> float:
    """
    训练模型一个 epoch。

    :param model: 模型。
    :param dataloader: 数据加载器。
    :param criterion: 损失函数。
    :param optimizer: 优化器。
    :param device: 设备 ('cuda' 或 'cpu')。
    :return: 平均训练损失。
    """

    model.train()

    total_loss = 0.0

    for x, y in dataloader:
        src = x.to(device)
        target = y.to(device)

        # 前向传播
        output = model(src)
        loss = criterion(output, target)

        # 反向传播
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()  # 计算梯度

        # 应用梯度裁剪
        clip_grad_norm_(model.parameters(), max_norm=1.0)  # 设置最大范数为1.0

        # 优化步骤
        optimizer.step()  # 更新参数

        total_loss += loss.item() * src.size(0)

    return total_loss / len(dataloader.dataset)  # type: ignore


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str | torch.device,
) -> tuple[float, WindowsData, WindowsData]:
    """
    评估模型。

    :param model: 模型。
    :param dataloader: 数据加载器。
    :param criterion: 损失函数。
    :param device: 设备 ('cuda' 或 'cpu')。
    :return: 测试损失、预测值和真实值。
    """

    model.eval()

    total_loss = 0.0
    pred = []
    truth = []

    with torch.no_grad():
        for x, y in dataloader:
            src = x.to(device)
            target = y.to(device)

            output = model(src)
            loss = criterion(output, target)

            total_loss += loss.item() * x.size(0)
            pred.append(output.cpu().numpy())
            truth.append(y.cpu().numpy())

    pred = np.concatenate(pred, axis=0)
    truth = np.concatenate(truth, axis=0)
    return total_loss / len(dataloader.dataset), pred, truth  # type: ignore


def run_experiment(
    *,
    model: nn.Module,
    train_dataset: Dataset,
    test_dataset: Dataset,
    run_times: int,
    num_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
) -> tuple[
    tuple[WindowsData, ...],
    WindowsData,
    tuple[float, ...],
    tuple[float, ...],
]:
    """
    运行实验, 进行多次训练和评估。

    :param output_window: 输出序列的长度。
    :param num_epochs: 训练的轮数。
    :param batch_size: 批量大小。
    :param learning_rate: 学习率。
    :param device: 设备 ('cuda' 或 'cpu')。
    :return: 预测值、真实值、MSE列表、MAE列表。
    """

    original_state = model.state_dict()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mse_list: list[float] = []
    mae_list: list[float] = []
    preds: list[WindowsData] = []

    device = torch.device(device)

    for run in range(run_times):
        print(f"Run {run + 1}/{run_times}")

        # 设置不同的随机种子以确保实验独立性
        set_seed(42 + run)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型
        model.load_state_dict(original_state)
        model = model.to(device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # 训练模型
        for epoch in range(1, num_epochs + 1):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{num_epochs}, Training Loss: {train_loss:.4f}")

        # 评估模型
        _test_loss, pred, truth = evaluate_model(model, test_loader, criterion, device)

        mse = cast("float", mean_squared_error(truth, pred))
        mae = cast("float", mean_absolute_error(truth, pred))

        print(f"Run {run + 1} - Test MSE: {mse:.4f}, Test MAE: {mae:.4f}\n")
        mse_list.append(mse)
        mae_list.append(mae)
        preds.append(pred)

    mse_avg = np.mean(mse_list)
    mse_std = np.std(mse_list)
    mae_avg = np.mean(mae_list)
    mae_std = np.std(mae_list)

    print(f"Average MSE: {mse_avg:.4f} ± {mse_std:.4f}")
    print(f"Average MAE: {mae_avg:.4f} ± {mae_std:.4f}")

    return tuple(preds), truth, tuple(mse_list), tuple(mae_list)
