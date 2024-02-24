import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load(path: str) -> pd.DataFrame:
    """Read a CSV datasheet and return a DataFrame."""
    df = pd.read_csv(path)
    return df

def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

def standard_scaler(data):
    mean = np.mean(data)
    scale = np.std(data - mean)
    return (data - mean) / scale

def export_thetas(parameters, str):
    thetas_dict = {}
    for values, house in parameters:
        thetas_dict[house] = values
    df = pd.DataFrame(thetas_dict)
    df.to_csv(f"../data_sets/parameters_{str}.csv", index=False)
    print(f"Exporting file : parameters has been saved to ./data_sets/parameters_{str}.csv")

def export_predict_house(class_predicted):
    df = pd.DataFrame(class_predicted, columns=['Hogwarts House'])
    df.to_csv("../data_sets/houses.csv", index_label='Index')
    print(f"Exporting file : houses has been saved to ./data_sets/houses.csv")
    return

def plot_cost(costs):
    for cost, house in costs:
        plt.plot(range(len(cost)), cost, label=f"{house}")

    plt.title("Convergence Graph of Cost Function for all Houses")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()

def get_terminal_size():
    """Permet d'obtenir la taille du terminal"""
    try:
        columns, rows = os.get_terminal_size(0)
        return columns, rows
    except OSError:
        print(OSError)
        return None

def ft_tqdm(lst: range) -> None:
    """Recoit une liste en argument et affiche une barre
     de progression en fonction du nombre de valeur dans la liste"""
    total = len(lst)
    terminal_size = get_terminal_size()

    for i, item in enumerate(lst):
        progress = min(1.0, (i + 1) / total)
        percentage = int(progress * 100)
        counter = f"{i + 1}/{total}"
        gap_size = terminal_size[0] - len(str(percentage)) - len(counter) - 30
        bar_length = int(progress * gap_size)
        bar = "█" * bar_length

        percentage_str = f"{percentage}%"
        bar_str = f"|{bar:<{gap_size}}|"
        progress_str = f"{i + 1}/{total}"

        output_str = f"{percentage_str}{bar_str} {progress_str}"

        print(f"\r{output_str}", end="", flush=True)
        yield item
    print()

def init_parameters(X):
    return np.random.randn(X.shape[1]) * 0.01
