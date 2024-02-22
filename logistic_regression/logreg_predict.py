import pandas as pd
import numpy as np
from utils import sigmoid_function, export_predict_house, load, standard_scaler


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
        X = data.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Arithmancy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Transfiguration', 'Potions', 'Care of Magical Creatures'], axis=1)
        X = X.fillna(0).values

        # Dataset de train predit pour le accuracy_score (remettre dataset test)
        data = load("../data_sets/dataset_train.csv")
        X_data = data.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name','Birthday', 'Best Hand', 'Arithmancy', 'Defense Against the Dark Arts', 'Divination', 'Transfiguration', 'Potions', 'Care of Magical Creatures'], axis=1)
        X_data = X_data.fillna(0).values
        X_data = standard_scaler(X_data)

        class_predicted = sorting_hat(X_data, thetas)
        export_predict_house(class_predicted)
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    predict()

if __name__ == "__main__":
    main()