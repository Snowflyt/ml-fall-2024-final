import numpy as np

from ml_fall_2024_final.constants import OutputWindowLong, OutputWindowShort

type WindowData = (
    np.ndarray[tuple[OutputWindowShort], np.dtype[np.float32]]
    | np.ndarray[tuple[OutputWindowLong], np.dtype[np.float32]]
)
type WindowsData = (
    np.ndarray[tuple[int, OutputWindowShort], np.dtype[np.float32]]
    | np.ndarray[tuple[int, OutputWindowLong], np.dtype[np.float32]]
)
