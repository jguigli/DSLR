import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def load(path: str) -> pd.DataFrame:
    """Read a CSV datasheet and return a DataFrame."""
    df = pd.read_csv(path)
    return df

def max(data):
    max = data[0]
    for value in data:
        if value > max:
            max = value
    return max

def min(data):
    min = data[0]
    for value in data:
        if value < min:
            min = value
    return min

def quantile(data, p):
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

def calcul_fields(numerical_data) -> pd.DataFrame():
    """"""
    try:
        i = 0
        describe_dataframe = pd.DataFrame(index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])

        for key, value in numerical_data.items():
            value_zero_nan = value.dropna()
            count = len(value_zero_nan)

            mean = np.sum(value_zero_nan) / count

            s_diff = [(element - mean) ** 2 for element in value_zero_nan]
            var = sum(s_diff) / count
            std = var ** 0.5

            mini = min(value_zero_nan)

            quartile1 = quantile(value_zero_nan, 0.25)
            quartile2 = quantile(value_zero_nan, 0.5)
            quartile3 = quantile(value_zero_nan, 0.75)

            maxi = max(value_zero_nan)

            median = sorted(value_zero_nan)[count // 2]

            # Rajouter files bonus : var, median, ...
            fields = [count, mean, std, mini, quartile1, quartile2, quartile3, maxi]
            fields_formatted = [f'{name:.6f}' for name in fields]

            describe_dataframe.insert(i, key, fields_formatted)
            i += 1
        return describe_dataframe
    except Exception as e:
        print(f"Error handling: {e}")

def describe():    
    try:
        data = load("../data_sets/dataset_train.csv")
        numerical_data = {}

        for name, column in data.items():
            if (column.dtypes == 'float64' or column.dtypes == 'int64'):
                numerical_data[name] = column
        print(data.describe())
        describe = calcul_fields(numerical_data)
        print(describe)
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    describe()

if __name__ == "__main__":
    main()