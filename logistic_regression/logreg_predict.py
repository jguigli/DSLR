import pandas as pd
import numpy as np
from utils import sigmoid_function, export_predict_house, load, standard_scaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def sorting_hat(X, parameters):
    X = np.insert(X, 0, 1, axis=1)
    final_house_predicted = []

    for value in X:
        probabilities = []

        for house, parameter in parameters.items():
            z = value.dot(parameter)
            probability = sigmoid_function(z)
            probabilities.append((probability, house))

        predicted_house = max(probabilities, key=lambda x: x[0])[1]
        final_house_predicted.append(predicted_house)
    return final_house_predicted

def predict():
    try:
        print("==> Sorting hat program <==\n")

        all_parameters = []
        parameters_batch = load("../data_sets/parameters_batch.csv")
        parameters_stochastic = load("../data_sets/parameters_stochastic.csv")
        parameters_minibatch = load("../data_sets/parameters_minibatch.csv")
        all_parameters.append((parameters_batch, "Batch"))
        all_parameters.append((parameters_stochastic, "Stochastic"))
        all_parameters.append((parameters_minibatch, "Mini Batch"))

        data = load("../data_sets/dataset_test.csv")
        X = data.drop(['Index',
                       'Hogwarts House',
                       'First Name',
                       'Last Name',
                       'Birthday',
                       'Best Hand',
                       'Arithmancy',
                       'Defense Against the Dark Arts',
                       'Transfiguration',
                       'Potions',
                       'Care of Magical Creatures'], axis=1)
        X = X.fillna(0).values
        X = standard_scaler(X)
        
        # Dataset contenant les instances corrects du dataset de test
        data_truth = load("../data_sets/dataset_truth.csv")
        y_truth = data_truth['Hogwarts House'].values

        # Affichage des scores de precisions des modeles en fonction de l'algortihme avec le dataset de test donnee
        print("Accuracy of the logistic regression model with the given test dataset :")
        for parameter, type in all_parameters:
            y_predicted = sorting_hat(X, parameter)
            if (type == "Batch"):
                y_predicted_mandatory = y_predicted
            accuracy = accuracy_score(y_truth, y_predicted)
            error = sum(y_predicted != y_truth)
            print(f"- {type} Gradient Descent algorithm : {(100 * accuracy):.2f}% ({error} error(s) out of {len(y_truth)} instances))")

        # Dataset de train pour creation de dataset test aleatoire avec train_test_split
        data = load("../data_sets/dataset_train.csv")
        y_data = data['Hogwarts House'].values
        X_data = data.drop(['Index',
                       'Hogwarts House',
                       'First Name',
                       'Last Name',
                       'Birthday',
                       'Best Hand',
                       'Arithmancy',
                       'Defense Against the Dark Arts',
                       'Transfiguration',
                       'Potions',
                       'Care of Magical Creatures'], axis=1)
        X_data = X_data.fillna(0).values
        X_data = standard_scaler(X_data)

        # Affichage des scores de precisions des modeles en fonction de l'algortihme avec des datasets aleatoires
        print("\nAccuracy of the logistic regression model with multiple random test datasets :")
        print("- Batch Gradient Descent algorithm :")
        total_error = 0
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X_data,y_data,test_size = 0.25)
            y_predicted = sorting_hat(X_test, parameters_batch)
            accuracy = accuracy_score(y_test, y_predicted)
            error = sum(y_predicted != y_test)
            total_error += error
            print(f"\t- Dataset test {i + 1} : {(100 * accuracy):.2f}% ({error} error(s) out of {len(y_test)} instances)")
        print(f"Average error rate : {round(total_error / 10)} error(s)")
        print("\n- Stochastic Gradient Descent algorithm :")
        total_error = 0
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X_data,y_data,test_size = 0.25)
            y_predicted = sorting_hat(X_test, parameters_stochastic)
            accuracy = accuracy_score(y_test, y_predicted)
            error = sum(y_predicted != y_test)
            total_error += error
            print(f"\t- Dataset test {i + 1} : {(100 * accuracy):.2f}% ({error} error(s) out of {len(y_test)} instances)")
        print(f"Average error rate : {round(total_error / 10)} error(s)")
        print("\n- Mini-Batch Gradient Descent algorithm :")
        total_error = 0
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X_data,y_data,test_size = 0.25)
            y_predicted = sorting_hat(X_test, parameters_batch)
            accuracy = accuracy_score(y_test, y_predicted)
            error = sum(y_predicted != y_test)
            total_error += error
            print(f"\t- Dataset test {i + 1} : {(100 * accuracy):.2f}% ({error} error(s) out of {len(y_test)} instances)")
        print(f"Average error rate : {round(total_error / 10)} error(s)")

        print()
        export_predict_house(y_predicted_mandatory)
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    predict()

if __name__ == "__main__":
    main()