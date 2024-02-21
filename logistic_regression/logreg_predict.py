import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import sigmoid_function, export_predict_house, load
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


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

def score(X, y, thetas): 
    # This function compares the predictd label with the actual label to find the model performance
    print(f"sum = {sorting_hat(X, thetas) == y}")
    print(f"len = {len(y)}")
    score = sum(sorting_hat(X, thetas) == y) / len(y)
    return score

def predict():
    try:
        data = load("../data_sets/dataset_test.csv")
        thetas = load("./thetas.csv")
        X = data.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Arithmancy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Transfiguration', 'Potions', 'Care of Magical Creatures'], axis=1)
        X = X.fillna(0).values

        data = load("../data_sets/dataset_train.csv")
        y_data = data['Hogwarts House'].values
        X_data = data.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Arithmancy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Transfiguration', 'Potions', 'Care of Magical Creatures'],axis=1)
        X_data = X_data.fillna(0).values

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.33)

        class_predicted = sorting_hat(X_test, thetas)
        export_predict_house(class_predicted)

        accuracy = accuracy_score(y_test, class_predicted)
        print(f"accuracy score = {round(100 * accuracy)}%\n")
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    predict()

if __name__ == "__main__":
    main()