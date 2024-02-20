import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import sigmoid_function, export_predict_house, load
from sklearn.metrics import accuracy_score


def sorting_hat(X, thetas):
    X = np.insert(X, 0, 1, axis=1)
    final_class_predict = []
    for value in X:
        probabilities = []
        for house, theta in thetas.items():
            z = value.dot(theta)
            probability = sigmoid_function(z)
            probabilities.append((probability, house))
        predicted_class = max(probabilities, key=lambda x: x[0])[1]
        final_class_predict.append(predicted_class)
    return final_class_predict


def predict():
    try:
        data = load("../data_sets/dataset_test.csv")
        thetas = load("./thetas.csv")
        X = data.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Arithmancy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures'],axis=1)
        
        # Met les valeurs NaN a 0
        X = X.fillna(0).values

        # X = standard_scaler(X)
        class_predicted = sorting_hat(X, thetas)
        export_predict_house(class_predicted)
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    predict()

if __name__ == "__main__":
    main()