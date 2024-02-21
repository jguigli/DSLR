import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load(path: str) -> pd.DataFrame:
    """Read a CSV datasheet and return a DataFrame."""
    df = pd.read_csv(path)
    return df

def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

def standard_scaler(data):
    mean = np.mean(data)
    scale = np.std(data - mean)
    return (data - mean) / scale

def export_thetas(thetas):
    thetas_dict = {}
    for values, house in thetas:
        thetas_dict[house] = values
    df = pd.DataFrame(thetas_dict)
    df.to_csv("./thetas.csv", index=False)
    print(f"Exporting file : thetas has been saved to /logistic_regression/thetas.csv")

def export_predict_house(class_predicted):
    df = pd.DataFrame(class_predicted, columns=['Hogwarts House'])
    df.to_csv("./houses.csv", index_label='Index')
    print(f"Exporting file : houses has been saved to /logistic_regression/houses.csv")
    return
