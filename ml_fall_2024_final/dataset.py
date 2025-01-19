from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd
import torch
from torch.utils.data import Dataset

from ml_fall_2024_final.constants import (
    INPUT_WINDOW,
    OUTPUT_WINDOW_LONG,
    OUTPUT_WINDOW_SHORT,
)


class BikeDataset(Dataset):
    """
    自定义数据集类, 用于生成输入序列和目标序列。

    :param data: 输入数据的 DataFrame。
    :param input_window: 输入序列的长度 (小时数)。
    :param output_window: 目标序列的长度 (小时数)。
    """

    features: ClassVar = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
    ]

    target: ClassVar = "cnt"

    def __init__(self, data: pd.DataFrame, /, *, output_window: Literal["short", "long"]) -> None:
        self.data = data.reset_index(drop=True)
        self.input_window = INPUT_WINDOW
        self.output_window = OUTPUT_WINDOW_SHORT if output_window == "short" else OUTPUT_WINDOW_LONG
        self.length = len(self.data) - self.input_window - self.output_window + 1

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data.loc[idx : idx + self.input_window - 1, BikeDataset.features].to_numpy()
        y = self.data.loc[
            idx + self.input_window : idx + self.input_window + self.output_window - 1,
            BikeDataset.target,
        ].to_numpy()
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def _get_data_path(filename: str) -> Path:
    """
    获取数据文件的绝对路径, 确保路径相对于当前文件所在位置。
    """
    return Path(__file__).resolve().parent.parent / "data" / filename


def read_train_data() -> pd.DataFrame:
    """
    读取训练数据。

    :return: 训练数据的 DataFrame。
    """
    data = pd.read_csv(_get_data_path("train_data.csv")).drop(
        columns=["instant", "casual", "registered"]
    )
    data["dteday"] = pd.to_datetime(data["dteday"], format="%Y/%m/%d")
    return data


def read_test_data() -> pd.DataFrame:
    """
    读取测试数据。

    :return: 测试数据的 DataFrame。
    """
    data = pd.read_csv(_get_data_path("test_data.csv")).drop(
        columns=["instant", "casual", "registered"]
    )
    data["dteday"] = pd.to_datetime(data["dteday"], format="%Y/%m/%d")
    return data
