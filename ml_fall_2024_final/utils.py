import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


def set_seed(seed: int, /):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 如果使用CuDNN, 可以设置以下选项以提高结果的可重复性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model: nn.Module, name: str, /) -> None:
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pt"
    torch.save(model.state_dict(), models_dir / filename)


def load_model(model: nn.Module, filename: str, /) -> nn.Module:
    models_dir = Path(__file__).resolve().parent.parent / "models"
    model.load_state_dict(torch.load(models_dir / filename))
    return model


def save_fig(fig: plt.Figure, name: str, /) -> None:  # type: ignore
    figures_dir = Path(__file__).resolve().parent.parent / "output" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
    fig.savefig(figures_dir / filename)
