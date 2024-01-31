import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load(path: str) -> pd.DataFrame:
    """Read a CSV datasheet and return a DataFrame."""
    df = pd.read_csv(path)
    return df

def calcul_fields(numerical_data) -> pd.DataFrame():
    """"""
    try:
        i = 0
        describe_dataframe = pd.DataFrame(index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'var', 'median'])

        for key, value in numerical_data.items():
            count = value.count()
            mean = value.mean()
            var = value.var()
            std = var ** 0.5
            min = value.min()
            quartile25 = value.quantile(0.25)
            quartile50 = value.quantile(0.5)
            quartile75 = value.quantile(0.75)
            max = value.max()
            median = value.median()

            fields = [count, mean, std, min, quartile25, quartile50, quartile75, max, var, median]
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
        # print(data.describe())
        describe = calcul_fields(numerical_data)
        print(describe)
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    describe()

if __name__ == "__main__":
    main()