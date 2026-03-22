import numpy as np
from numpy.typing import NDArray


def moving_average(data: NDArray, window_size: int) -> NDArray[np.floating]:
    if len(data) < window_size:
        return np.full(len(data), np.mean(data))
    weights = np.ones(window_size) / window_size
    out = np.convolve(data, weights, mode="same")
    k = len(weights) // 2
    out[:k] = data[k]
    out[-k:] = data[-k]
    return out


def small_run_filter(data: NDArray[np.bool], min_size: int) -> NDArray[np.bool]:
    borders = np.flatnonzero(np.diff(data)) + 1
    if len(borders) == 0:
        return data.copy()
    # borders = np.concatenate(([0], borders, [len(data)]))
    out = data.copy()
    for start, end in zip(borders, borders[1:]):
        if (end - start) <= min_size:
            if start == 0:
                out[start:end] = out[end]
                continue
            out[start:end] = out[start - 1]
    return out


def apply_kernel(data: NDArray, kernel: NDArray) -> NDArray[np.floating]:
    kernel_len = len(kernel)
    if (kernel_len % 2) == 0:
        raise ValueError("kernel must have an odd length")
    if len(data) < kernel_len:
        return data.astype(np.float64)
    out = data.copy()
    buffer = int((kernel_len - 1) / 2)
    out[buffer:-buffer] = np.convolve(data, kernel, mode="valid")
    assert len(data) == len(out)
    return out


def square_filter(data: NDArray, step_up: bool) -> NDArray[np.floating]:
    if data.dtype == np.bool:
        data = data.astype(np.int16)
    step = [1, -1] if step_up else [-1, 1]
    normed = normalize(data, *step)

    sums = np.cumsum(normed)
    total = sums[-1]

    return (2 * sums - total) / len(data)


def normalize(data: NDArray, min_: float | np.number, max_: float | np.number) -> NDArray:

    scaler = ((data.max() - data.min()) / (max_ - min_)) + 1e-9
    move = ((max_ + min_) / 2) - (((data.max() + data.min()) / 2) / scaler)

    return (data / scaler) + move
