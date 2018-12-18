"""Test implementations of softmax functions.

References:
- https://en.wikipedia.org/wiki/Softmax_function
- https://stackoverflow.com/q/9906136
"""

import math
from typing import Collection, List, Union
import numpy as np
import scipy.special as ss
import torch


Number = Union[int, float]

_VAL_UNDERFLOW = -746
_VAL_SMALL = -745
_VAL_NORMAL = 1
_VAL_LARGE = 709
_VAL_OVERFLOW = 710


def naive_math(x: Collection[Number]) -> List[float]:
    x_exp = [math.exp(i) for i in x]
    sum_x_exp = sum(x_exp)
    _softmax = [i / sum_x_exp for i in x_exp]
    return _softmax


def naive_numpy(x: Union[Collection[Number], np.ndarray]) -> np.ndarray:
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp)


def naive_pytorch(x: Union[Collection[Number], np.ndarray]) -> torch.tensor:
    """https://pytorch.org/docs/stable/nn.html#torch.nn.Softmax"""
    return torch.nn.Softmax(dim=0)(torch.tensor(x))


def log_pytorch(x: Union[Collection[Number], np.ndarray]) -> torch.tensor:
    """
    https://pytorch.org/docs/stable/nn.html#logsoftmax
    https://pytorch.org/docs/stable/tensors.html#torch.Tensor.exp
    https://pytorch.org/docs/stable/torch.html#torch.exp
    """
    return torch.nn.LogSoftmax(dim=0)(torch.tensor(x)).exp()


def scipy(x: Union[Collection[Number], np.ndarray]) -> np.ndarray:
    """https://stackoverflow.com/a/48815016/"""
    return np.exp(x - ss.logsumexp(x))


_WELL_BEHAVED_INPUT = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
_WELL_BEHAVED_OUTPUT = [
    0.023640543021591385,
    0.06426165851049616,
    0.17468129859572226,
    0.4748329997443803,
    0.023640543021591385,
    0.06426165851049616,
    0.17468129859572226
]


def test_naive_math() -> None:
    output = naive_math(_WELL_BEHAVED_INPUT)
    np.testing.assert_allclose(output, _WELL_BEHAVED_OUTPUT, rtol=0, atol=1.0e-9)
    return


def test_naive_numpy() -> None:
    output = naive_numpy(_WELL_BEHAVED_INPUT)
    np.testing.assert_allclose(output, _WELL_BEHAVED_OUTPUT, rtol=0, atol=1.0e-9)
    return


def test_naive_pytorch() -> None:
    output = naive_pytorch(_WELL_BEHAVED_INPUT).numpy()
    np.testing.assert_allclose(output, _WELL_BEHAVED_OUTPUT, rtol=0, atol=7.5e-8)
    return


def test_log_pytorch() -> None:
    output = log_pytorch(_WELL_BEHAVED_INPUT).numpy()
    np.testing.assert_allclose(output, _WELL_BEHAVED_OUTPUT, rtol=0, atol=7.5e-8)
    return


def test_scipy() -> None:
    output = scipy(_WELL_BEHAVED_INPUT)
    np.testing.assert_allclose(output, _WELL_BEHAVED_OUTPUT, rtol=0, atol=1.0e-9)
    return


if __name__ == "__main__":
    print(naive_math(_WELL_BEHAVED_INPUT))
    print(naive_numpy(_WELL_BEHAVED_INPUT))
    print(naive_pytorch(_WELL_BEHAVED_INPUT).numpy())
    print(log_pytorch(_WELL_BEHAVED_INPUT).numpy())
    print(scipy(_WELL_BEHAVED_INPUT))
