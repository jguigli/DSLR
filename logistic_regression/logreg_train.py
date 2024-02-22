import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import sigmoid_function, standard_scaler, load, export_thetas


def gradient_descent(X, y):
    m = len(y)
    learning_rate = 1
    thetas = []
    costs = []

    # Rajout d'une colonne pour le biais
    X = np.insert(X, 0, 1, axis=1)

    for house in np.unique(y):
        # Mise a jour des etiquettes des donnees y correpondantes par 1 et 0 pour ceux qui ne sont pas concernees
        y_current = np.where(y == house, 1, 0)
        # Creation d'un vecteur theta a l'echelle du nombres de colonnes des features
        theta = np.zeros(X.shape[1])
        cost = []
        for _ in range(50000):
            z = X.dot(theta)
            h = sigmoid_function(z)
            
            gradient_value = 1 / m * np.dot((h - y_current), X)
            theta -= learning_rate * gradient_value
        thetas.append((theta, house))
    return thetas

def train_model():
    try:
        data = load("../data_sets/dataset_train.csv")
        y = data['Hogwarts House'].values
        X = data.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Arithmancy', 'Defense Against the Dark Arts', 'Divination', 'Transfiguration', 'Potions', 'Care of Magical Creatures'], axis=1)
        
        # Met les valeurs NaN a 0
        X = X.fillna(0).values
        X = standard_scaler(X)

        thetas = gradient_descent(X, y)
        export_thetas(thetas)
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    train_model()

if __name__ == "__main__":
    main()