from collections.abc import Sequence
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ml_fall_2024_final.type import WindowData, WindowsData


def print_window_mse_mae(preds: Sequence[WindowData], truth: WindowsData, /) -> None:
    """
    打印窗口的 MSE/MAE。

    :param preds: 预测值。
    :param truth: 真实值。
    """

    mse_list: list[float] = []
    mae_list: list[float] = []

    for pred in preds:
        mse = cast("float", mean_squared_error(truth, pred))
        mae = cast("float", mean_absolute_error(truth, pred))
        mse_list.append(mse)
        mae_list.append(mae)

    mse_avg = np.mean(mse_list)
    mse_std = np.std(mse_list)
    mae_avg = np.mean(mae_list)
    mae_std = np.std(mae_list)

    print(f"Average MSE: {mse_avg:.4f} ± {mse_std:.4f}")
    print(f"Average MAE: {mae_avg:.4f} ± {mae_std:.4f}")


def plot_window_prediction(
    pred_or_preds: WindowData | Sequence[WindowData],
    truth: WindowData,
    /,
    *,
    title: str = "Predictions vs Ground Truth",
    figsize: tuple[int, int] = (15, 5),
) -> plt.Figure:  # type: ignore
    """
    绘制预测值与真实值的对比图。

    :param prediction: 预测值。
    :param truth: 真实值。
    :param title: 图表标题。
    """

    fig = plt.figure(figsize=figsize)

    if isinstance(pred_or_preds, np.ndarray):
        plt.plot(pred_or_preds, label="Prediction")
        plt.plot(truth, label="Ground Truth")
    else:
        plt.plot(truth, label="Ground Truth", color="blue")
        for i, pred in enumerate(pred_or_preds):
            plt.plot(pred, label=f"Prediction {i + 1}", color="orange", linestyle="--")
    plt.xlabel("Hour")
    plt.ylabel("Bike Rentals (cnt)")
    plt.title(title)
    plt.legend()

    return fig
