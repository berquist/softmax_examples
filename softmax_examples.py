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
    return torch.nn.Softmax(dim=0)(torch.tensor(x).double())


def log_pytorch(x: Union[Collection[Number], np.ndarray]) -> torch.tensor:
    """
    https://pytorch.org/docs/stable/nn.html#logsoftmax
    https://pytorch.org/docs/stable/tensors.html#torch.Tensor.exp
    https://pytorch.org/docs/stable/torch.html#torch.exp
    """
    return torch.nn.LogSoftmax(dim=0)(torch.tensor(x).double()).exp()


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
    0.17468129859572226,
]

_UNDERFLOW_INPUT = [
    _VAL_UNDERFLOW,
    _VAL_UNDERFLOW + 1,
    _VAL_UNDERFLOW + 2,
    _VAL_UNDERFLOW + 3,
    _VAL_UNDERFLOW + 10,
]
_UNDERFLOW_BAD_OUTPUT = [
    0.0,
    0.00021570319240724764,
    0.0004314063848144953,
    0.0008628127696289905,
    0.9984900776531492,
]
_UNDERFLOW_GOOD_OUTPUT = [
    4.53357274e-05,
    1.23235284e-04,
    3.34988233e-04,
    9.10592426e-04,
    9.98585848e-01,
]


def test_naive_math() -> None:
    well_behaved_output = naive_math(_WELL_BEHAVED_INPUT)
    np.testing.assert_allclose(well_behaved_output, _WELL_BEHAVED_OUTPUT, rtol=0, atol=1.0e-9)
    underflow_output = naive_math(_UNDERFLOW_INPUT)
    np.testing.assert_allclose(underflow_output, _UNDERFLOW_BAD_OUTPUT, rtol=0, atol=1.0e-9)
    return


def test_naive_numpy() -> None:
    well_behaved_output = naive_numpy(_WELL_BEHAVED_INPUT)
    np.testing.assert_allclose(well_behaved_output, _WELL_BEHAVED_OUTPUT, rtol=0, atol=1.0e-9)
    underflow_output = naive_numpy(_UNDERFLOW_INPUT)
    np.testing.assert_allclose(underflow_output, _UNDERFLOW_BAD_OUTPUT, rtol=0, atol=1.0e-9)
    return


def test_naive_pytorch() -> None:
    well_behaved_output = naive_pytorch(_WELL_BEHAVED_INPUT).numpy()
    np.testing.assert_allclose(well_behaved_output, _WELL_BEHAVED_OUTPUT, rtol=0, atol=7.5e-8)
    # PyTorch is safe to use.
    underflow_output = naive_pytorch(_UNDERFLOW_INPUT).numpy()
    np.testing.assert_allclose(underflow_output, _UNDERFLOW_GOOD_OUTPUT, rtol=0, atol=7.5e-8)
    return


def test_log_pytorch() -> None:
    well_behaved_output = log_pytorch(_WELL_BEHAVED_INPUT).numpy()
    np.testing.assert_allclose(well_behaved_output, _WELL_BEHAVED_OUTPUT, rtol=0, atol=7.5e-8)
    # PyTorch is safe to use.
    underflow_output = log_pytorch(_UNDERFLOW_INPUT).numpy()
    np.testing.assert_allclose(underflow_output, _UNDERFLOW_GOOD_OUTPUT, rtol=0, atol=7.5e-8)
    return


def test_scipy() -> None:
    well_behaved_output = scipy(_WELL_BEHAVED_INPUT)
    np.testing.assert_allclose(well_behaved_output, _WELL_BEHAVED_OUTPUT, rtol=0, atol=1.0e-9)
    # SciPy is safe to use.
    underflow_output = scipy(_UNDERFLOW_INPUT)
    np.testing.assert_allclose(underflow_output, _UNDERFLOW_GOOD_OUTPUT, rtol=0, atol=1.0e-9)
    return


if __name__ == "__main__":
    print(naive_math(_WELL_BEHAVED_INPUT))
    print(naive_numpy(_WELL_BEHAVED_INPUT))
    print(naive_pytorch(_WELL_BEHAVED_INPUT).numpy())
    print(log_pytorch(_WELL_BEHAVED_INPUT).numpy())
    print(scipy(_WELL_BEHAVED_INPUT))
