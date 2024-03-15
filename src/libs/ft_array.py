import numpy as np


class ft_array:
    """class ft_array for array operations and statistics."""

    @staticmethod
    def describe(arr: np.ndarray):
        return {
            "count": ft_array.count(arr),
            "mean": ft_array.mean(arr),
            "std": ft_array.std(arr),
            "min": ft_array.min(arr),
            "25%": ft_array.percentile(arr, 25),
            "50%": ft_array.percentile(arr, 50),
            "75%": ft_array.percentile(arr, 75),
            "max": ft_array.max(arr),
        }

    @staticmethod
    def count(arr: np.ndarray):
        return len(arr)

    @staticmethod
    def mean(arr: np.ndarray):
        return np.sum(arr) / len(arr)

    @staticmethod
    def std(arr: np.ndarray):
        return (np.sum((arr - ft_array.mean(arr)) ** 2) / len(arr)) ** 0.5

    @staticmethod
    def min(arr: np.ndarray):
        _min = arr[0]
        for i in range(1, len(arr)):
            if arr[i] < _min:
                _min = arr[i]
        return _min

    @staticmethod
    def max(arr: np.ndarray):
        _max = arr[0]
        for i in range(1, len(arr)):
            if arr[i] > _max:
                _max = arr[i]
        return _max

    @staticmethod
    def percentile(arr: np.ndarray, p: int):
        sorted_arr = np.sort(arr)
        count = len(arr)
        position = (p / 100) * (count - 1)
        return np.interp(position, np.arange(count), sorted_arr)
