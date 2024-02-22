import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import load
from sklearn.metrics import accuracy_score


def score(y_pred, y_true): 
    # This function compares the predictd label with the actual label to find the model performance
    print(sum(y_pred == y_true))
    print(len(y_true))
    score = sum(y_pred == y_true) / len(y_true)
    return score

def predict():
    try:
        data_pred = load("./houses.csv")
        y_pred = data_pred['Hogwarts House'].values

        data = load("../data_sets/dataset_train.csv")
        y_true = data['Hogwarts House'].values

        # accuracy = accuracy_score(y_test, class_predicted)
        accuracy = score(y_pred, y_true)
        print(f"accuracy score = {(100 * accuracy):.2f}%\n")
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    predict()

if __name__ == "__main__":
    main()