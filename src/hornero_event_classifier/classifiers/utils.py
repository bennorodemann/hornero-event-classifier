"""This module contains a few util functions. None of them are currently being used but have been left in for future availability
if needed."""

import numpy as np
from numpy.typing import NDArray


def moving_average(data: NDArray, window_size: int) -> NDArray[np.floating]:
    """Compute a centered moving average.

    If the input length is smaller than ``window_size``, the output is a constant array filled with the mean of ``data``.

    :param data: Input array.
    :type data: NDArray
    :param window_size: Size of the moving average window.
    :type window_size: int
    :return: Smoothed array of the same length as ``data``.
    :rtype: NDArray[np.floating]
    """
    # if the length of the data is smaller than the window size return an array of the same size with the mean of all values
    if len(data) < window_size:
        return np.full(len(data), np.mean(data))
    # apply moving average
    weights = np.ones(window_size) / window_size
    out = np.convolve(data, weights, mode="same")
    # replace half of window size on each side with original valuies
    k = len(weights) // 2
    out[:k] = data[k]
    out[-k:] = data[-k]
    return out


def small_run_filter(data: NDArray[np.bool], min_size: int) -> NDArray[np.bool]:
    """Remove short runs of constant values by merging them into their neighbors.

    :param data: Input boolean array.
    :type data: NDArray[np.bool]
    :param min_size: Minimum run length to keep.
    :type min_size: int
    :return: Filtered boolean array.
    :rtype: NDArray[np.bool]
    """
    # find indexes where values switch (True -> False or False -> True)
    borders = np.flatnonzero(np.diff(data)) + 1
    # if never switch return copy unchanged
    if len(borders) == 0:
        return data.copy()
    out = data.copy()
    # compare each boundary with the next
    for start, end in zip(borders, borders[1:]):
        # if the difference is beyond the minimum size overwrite the values
        if (end - start) <= min_size:
            if start == 0:
                out[start:end] = out[end]
            else:
                out[start:end] = out[start - 1]
    return out


def apply_kernel(data: NDArray, kernel: NDArray) -> NDArray[np.floating]:
    """Apply a 1D kernel to the center of an array.

    The kernel must have odd length. If ``data`` is shorter than the kernel, the input is returned as ``float``.

    :param data: Input array.
    :type data: NDArray
    :param kernel: Kernel to apply.
    :type kernel: NDArray
    :return: Filtered array.
    :rtype: NDArray[np.floating]
    :raises ValueError: If ``kernel`` has even length.
    """
    kernel_len = len(kernel)
    # require kernel to have an odd length
    if (kernel_len % 2) == 0:
        raise ValueError("kernel must have an odd length")
    # return data as float if length is < kernel length
    if len(data) < kernel_len:
        return data.astype(np.float64)
    out = data.copy()
    # replace center of array with applied kernel array
    buffer = int((kernel_len - 1) / 2)
    out[buffer:-buffer] = np.convolve(data, kernel, mode="valid")
    assert len(data) == len(out)
    return out


def square_filter(data: NDArray, step_up: bool) -> NDArray[np.floating]:
    """Compute a square-wave fit score across the array.

    :param data: Input array.
    :type data: NDArray
    :param step_up: If ``True``, model a step up then down. If ``False``, step down then up.
    :type step_up: bool
    :return: Score per index indicating best step boundary (higher is better).
    :rtype: NDArray[np.floating]
    """
    # convert to int if currently a bool
    if data.dtype == np.bool:
        data = data.astype(np.int16)
    # determine the direction of the square filter
    step = [1, -1] if step_up else [-1, 1]
    # scale data to be between -1 and 1
    normed = normalize(data, *step)

    sums = np.cumsum(normed)
    total = sums[-1]
    # returns score of best fit, index of max value is best boundary
    return (2 * sums - total) / len(data)


def normalize(data: NDArray, min_: float | np.number, max_: float | np.number) -> NDArray:
    """Linearly rescale ``data`` to the range ``[min_, max_]``.

    :param data: Input array.
    :type data: NDArray
    :param min_: Desired minimum value.
    :type min_: float | np.number
    :param max_: Desired maximum value.
    :type max_: float | np.number
    :return: Rescaled array.
    :rtype: NDArray
    """
    scaler = ((data.max() - data.min()) / (max_ - min_)) + 1e-9
    move = ((max_ + min_) / 2) - (((data.max() + data.min()) / 2) / scaler)

    return (data / scaler) + move
