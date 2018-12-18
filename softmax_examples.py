"""Test implementations of softmax functions.

References:
- https://en.wikipedia.org/wiki/Softmax_function
- https://stackoverflow.com/q/9906136
"""

import math
from typing import Collection, List, Union
import numpy as np
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


def naive_numpy(x: np.ndarray) -> np.ndarray:
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp)


def naive_pytorch(x):
    return torch.nn.Softmax()(torch.tensor(x))


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


def test_naive_math():
    output = naive_math(_WELL_BEHAVED_INPUT)
    np.testing.assert_allclose(output, _WELL_BEHAVED_OUTPUT)
    return


def test_naive_numpy():
    output = naive_numpy(_WELL_BEHAVED_INPUT)
    np.testing.assert_allclose(output, _WELL_BEHAVED_OUTPUT)
    return


def test_naive_pytorch():
    output = naive_pytorch(_WELL_BEHAVED_INPUT).numpy()
    np.testing.assert_allclose(output, _WELL_BEHAVED_OUTPUT)
    return
