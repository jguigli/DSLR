import sys
import numpy as np
import time
from .utils import sigmoid_function, standard_scaler, load, export_thetas, ft_tqdm, plot_cost, init_parameters


def gradient_descent(X, y):
    m = len(y)
    learning_rate = 1
    parameters = []
    costs = []

    # Rajout d'une colonne pour le biais
    X = np.insert(X, 0, 1, axis=1)
    start = time.time()
    for house in np.unique(y):
        y_current = np.where(y == house, 1, 0)
        parameter = init_parameters(X)
        cost = []
        for _ in ft_tqdm(range(50000)):
            batch_indices = np.random.choice(len(X), size=32, replace=False)
            X_batch = X[batch_indices]
            y_batch = y_current[batch_indices]

            z = X_batch.dot(parameter)
            h = sigmoid_function(z)
            
            gradient_value = 1 / m * np.dot((h - y_batch), X_batch)
            parameter -= learning_rate * gradient_value
            cost_value = -1 / m * np.sum(y_batch * np.log(h) + (1 - y_batch) * np.log(1 - h))
            cost.append(cost_value)
        parameters.append((parameter, house))
        costs.append((cost, house))
        print(f"Mini Batch Gradient descent has finished for the {house} label\n")
    end = time.time()
    print(f"Training of the model carried out in : {(end - start):.1f} second(s)\n")
    return parameters, costs

def train_model(path: str):
    try:
        print("===> Training of the logistic regression model <===\n")
        print("Algorithm : Mini Batch Gradient Descent\n")
        data = load(path)
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
        export_thetas(parameters, "minibatch")
        plot_cost(costs)
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main(*args):
    if args is None or len(args) != 2:
        print("Usage: python -m logreg_train_minibatch <path_to_dataset>")
        return
    train_model(args[1])

if __name__ == "__main__":
    main(*sys.argv[:])