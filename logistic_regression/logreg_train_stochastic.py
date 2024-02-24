import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from utils import sigmoid_function, standard_scaler, load, export_thetas, ft_tqdm, plot_cost, init_parameters


def gradient_descent(X, y):
    m = len(y)
    learning_rate = 1
    parameters = []
    costs = []

    # Rajout d'une colonne pour le biais
    X = np.insert(X, 0, 1, axis=1)
    start = time.time()
    for house in np.unique(y):
        # Mise a jour des etiquettes des donnees y correpondantes par 1 et 0 pour ceux qui ne sont pas concernees
        y_current = np.where(y == house, 1, 0)
        # Creation d'un vecteur parameter a l'echelle du nombres de colonnes des features
        parameter = init_parameters(X)
        cost = []
        for _ in ft_tqdm(range(50000)):
            index = np.random.randint(m)
            x_rand = X[index]
            y_rand = y_current[index]

            z = x_rand.dot(parameter)
            h = sigmoid_function(z)
            
            gradient_value = 1 / m * np.dot((h - y_rand), x_rand)
            parameter -= learning_rate * gradient_value
            cost_value = -1 / m * np.sum(y_rand * np.log(h) + (1 - y_rand) * np.log(1 - h))
            cost.append(cost_value)
        parameters.append((parameter, house))
        costs.append((cost, house))
        print(f"Stochastic Gradient descent has finished for the {house} label\n")
    end = time.time()
    print(f"Training of the model carried out in : {(end - start):.1f} second(s)\n")
    return parameters, costs

def train_model():
    try:
        print("===> Training of the logistic regression model <===\n")
        print("Algorithm : Stochastic Gradient Descent\n")
        data = load("../data_sets/dataset_train.csv")
        y = data['Hogwarts House'].values
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

        parameters, costs = gradient_descent(X, y)
        export_thetas(parameters, "stochastic")
        plot_cost(costs)
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    train_model()

if __name__ == "__main__":
    main()