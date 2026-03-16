"""
Simplified utility functions for PPO.
Based on stable_baselines3 but simplified for basic PPO usage.
"""

from collections import deque
from typing import Callable, Union
import numpy as np
import torch as th


# Type alias for schedule functions
Schedule = Callable[[float], float]


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes fraction of variance that y_pred explains about y_true.
    Returns 1 - Var[y-ypred] / Var[y]

    Interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: Prediction
    :param y_true: Expected value
    :return: Explained variance
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)


def safe_mean(arr: Union[np.ndarray, list, deque]) -> float:
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. Used for logging only.

    :param arr: Numpy array or list of values
    :return: Mean value or NaN if empty
    """
    return np.nan if len(arr) == 0 else float(np.mean(arr))


def obs_as_tensor(obs: Union[np.ndarray, dict[str, np.ndarray]], device: th.device) -> Union[th.Tensor, dict[str, th.Tensor]]:
    """
    Convert observation to PyTorch tensor on the specified device.

    :param obs: Observation (numpy array or dict of arrays)
    :param device: PyTorch device
    :return: PyTorch tensor or dict of tensors
    """
    if isinstance(obs, np.ndarray):
        return th.as_tensor(obs, device=device)
    elif isinstance(obs, dict):
        return {key: th.as_tensor(_obs, device=device) for key, _obs in obs.items()}
    else:
        raise TypeError(f"Unrecognized type of observation {type(obs)}")


def update_learning_rate(optimizer: th.optim.Optimizer, learning_rate: float) -> None:
    """
    Update the learning rate for the optimizer.

    :param optimizer: PyTorch optimizer
    :param learning_rate: New learning rate value
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


class ConstantSchedule:
    """
    Constant schedule that always returns the same value.

    :param val: Constant value
    """

    def __init__(self, val: float):
        self.val = val

    def __call__(self, _: float) -> float:
        return self.val

    def __repr__(self) -> str:
        return f"ConstantSchedule(val={self.val})"


class LinearSchedule:
    """
    Linear schedule that interpolates between start and end values.

    :param start: Starting value (when progress_remaining = 1)
    :param end: Ending value (when progress_remaining = 0)
    :param end_fraction: Fraction of training where end value is reached
    """

    def __init__(self, start: float, end: float, end_fraction: float = 1.0):
        self.start = start
        self.end = end
        self.end_fraction = end_fraction

    def __call__(self, progress_remaining: float) -> float:
        if (1 - progress_remaining) > self.end_fraction:
            return self.end
        else:
            return self.start + (1 - progress_remaining) * (self.end - self.start) / self.end_fraction

    def __repr__(self) -> str:
        return f"LinearSchedule(start={self.start}, end={self.end}, end_fraction={self.end_fraction})"


class FloatSchedule:
    """
    Wrapper that ensures the output of a Schedule is cast to float.
    Can wrap either a constant value or a callable Schedule.

    :param value_schedule: Constant value or callable schedule
    """

    def __init__(self, value_schedule: Union[Schedule, float]):
        if isinstance(value_schedule, FloatSchedule):
            self.value_schedule: Schedule = value_schedule.value_schedule
        elif isinstance(value_schedule, (float, int)):
            self.value_schedule = ConstantSchedule(float(value_schedule))
        else:
            assert callable(value_schedule), f"The schedule must be a float or callable, not {value_schedule}"
            self.value_schedule = value_schedule

    def __call__(self, progress_remaining: float) -> float:
        return float(self.value_schedule(progress_remaining))

    def __repr__(self) -> str:
        return f"FloatSchedule({self.value_schedule})"


def get_device(device: Union[th.device, str] = "auto") -> th.device:
    """
    Get the PyTorch device.

    :param device: Device name ('auto', 'cpu', 'cuda', etc.)
    :return: PyTorch device
    """
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"

    return th.device(device)


def set_random_seed(seed: int | None = None) -> None:
    """
    Set random seed for reproducibility.

    :param seed: Random seed
    """
    if seed is not None:
        np.random.seed(seed)
        th.manual_seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed(seed)
            th.cuda.manual_seed_all(seed)
