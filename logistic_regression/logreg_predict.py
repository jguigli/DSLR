import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import sigmoid_function, standard_scaler, load

def export_predict_house(class_predicted):
    df = pd.DataFrame(class_predicted)
    df.to_csv("./house.csv")
    print(f"Exporting file : house.csv has been saved to /logistic_regression")
    return

def sorting_hat(X, thetas):
    X = np.insert(X, 0, 1, axis=1)
    final_class_predict = []
    for value in X:
        probabilities = []
        for house, theta in thetas.items():
            # Peut etre a tranposer ?
            z = value.T.dot(theta)
            probability = sigmoid_function(z)
            probabilities.append((probability, house))
        
        predicted_class = max(probabilities, key=lambda x: x[0])[1]
        final_class_predict.append(predicted_class)
    return final_class_predict


def predict():
    try:
        data = load("../data_sets/dataset_test.csv")
        thetas = load("./thetas.csv")
        y = data['Hogwarts House'].values
        X = data.drop(['Index','Hogwarts House','First Name','Last Name','Birthday','Best Hand','Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying'],axis=1)
        # Supprime les valeurs NaN
        X = X.dropna()
        # Met a jour le nombres d'index de y pour correspondre a X
        y = y[X.index]
        X = X.values

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