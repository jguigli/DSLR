import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import sigmoid_function, standard_scaler, load, export_thetas

def compare_thetas(theta, old_theta):
    for i in range(len(theta)):
        if old_theta[i] != theta[i]:
            return False
    return True

def gradient_descent(X, y):
    m = len(y)
    learning_rate = 0.01
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
        for _ in range(30000):
            old_theta = theta
            z = X.dot(theta)
            h = sigmoid_function(z)
            
            gradient_value = 1 / m * np.dot(X.T, (h - y_current))
            theta -= learning_rate * gradient_value

            if (compare_thetas(theta, old_theta)):
                break
        thetas.append((theta, house))
    return thetas

def train_model():
    try:
        data = load("../data_sets/dataset_train.csv")
        y = data['Hogwarts House'].values
        X = data.drop(['Index','Hogwarts House','First Name','Last Name','Birthday','Best Hand','Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying'],axis=1)
        # Supprime les valeurs NaN
        X = X.dropna()
        # Met a jour le nombres d'index de y pour correspondre a X
        y = y[X.index]

        X = X.values
        # print(X.shape)
        # print(X)
        # print(y.shape)
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