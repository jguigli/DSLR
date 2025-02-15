import pandas as pd
import numpy as np
from ..libs import CSV
import sys



def ft_mode(data):
    counts = {}
    for value in data:
        counts[value] = counts.get(value, 0) + 1
    max_count = max(counts.values())
    mode = [key for key, count in counts.items() if count == max_count]
    return mode[0]


def ft_skewness(data):
    n = len(data)
    mean = sum(data) / n
    numerator = sum((x - mean) ** 3 for x in data)
    denominator = (sum((x - mean) ** 2 for x in data) / n) ** (3 / 2)
    return numerator / denominator


def ft_kurtosis(data):
    n = len(data)
    mean = sum(data) / n
    numerator = sum((x - mean) ** 4 for x in data)
    denominator = (sum((x - mean) ** 2 for x in data) / n) ** 2
    return (numerator / denominator) - 3


def ft_max(data):
    max = data[0]
    for value in data:
        if value > max:
            max = value
    return max


def ft_min(data):
    min = data[0]
    for value in data:
        if value < min:
            min = value
    return min


def ft_quantile(data, p):
    data = sorted(data)
    position = (len(data) - 1) * p
    floor = np.floor(position)
    ceil = np.ceil(position)

    if floor == ceil:
        return data[int(position)]

    # Linear interpolation
    d0 = data[int(floor)] * (ceil - position)
    d1 = data[int(ceil)] * (position - floor)
    return d0 + d1


def calcul_fields(numerical_data) -> pd.DataFrame:
    """"""
    try:
        i = 0
        describe_dataframe = pd.DataFrame(
            index=[
                "count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
                "nan",
                "mode",
                "skewness",
                "kurtosis",
            ]
        )

        for key, value in numerical_data.items():
            value_zero_nan = value.dropna()
            count = len(value_zero_nan)

            mean = np.sum(value_zero_nan) / count

            deviation = np.abs(value_zero_nan - mean) ** 2
            variance = sum(deviation) / (len(value_zero_nan) - 1)
            std = np.sqrt(variance)

            quartile1 = ft_quantile(value_zero_nan, 0.25)
            quartile2 = ft_quantile(value_zero_nan, 0.5)
            quartile3 = ft_quantile(value_zero_nan, 0.75)
            mini = ft_min(value_zero_nan)
            maxi = ft_max(value_zero_nan)

            nan = len(value) - count
            mode = ft_mode(value_zero_nan)
            skewness = ft_skewness(value_zero_nan)
            kurtosis = ft_kurtosis(value_zero_nan)

            # Rajouter files bonus : var, median, ...
            fields = [
                count,
                mean,
                std,
                mini,
                quartile1,
                quartile2,
                quartile3,
                maxi,
                nan,
                mode,
                skewness,
                kurtosis,
            ]
            fields_formatted = [f"{name:.6f}" for name in fields]

            describe_dataframe.insert(i, key, fields_formatted)
            i += 1
        return describe_dataframe
    except Exception as e:
        print(f"Error handling: {e}")


def describe(path: str):
    try:
        data = CSV.load(path)
        numerical_data = {}

        for name, column in data.items():
            if column.dtypes == "float64" or column.dtypes == "int64":
                numerical_data[name] = column
        print("DataFrame.describe() :")
        print(data.describe())
        describe = calcul_fields(numerical_data)
        print("\nFt_describe() :")
        print(describe)
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return


def main(*args):
    if args is None or len(args) != 2:
        print("Usage: python -m describe <dataset>")
        return
    describe(args[1])


if __name__ == "__main__":
    main(*sys.argv[:])
