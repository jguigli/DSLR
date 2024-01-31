import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load(path: str) -> pd.DataFrame:
    """Read a CSV datasheet and return a DataFrame."""
    df = pd.read_csv(path)
    return df

def scatter_data():    
    try:
        data = load("../data_sets/dataset_train.csv")
        data_x = data['Astronomy']
        data_y = data['Ancient Runes']

        # colors = ['cyan', 'purple']
        colors = np.random.rand(1600)

        plt.scatter(data_x, data_y, c=colors)
        # plt.plot([min(mileage), max(mileage)], [max(predicted_price), min(predicted_price)], color='red')
        plt.title('What are the two features that are similar ?')
        plt.xlabel('Astronomy')
        plt.ylabel('Ancient Runes')
        plt.show()
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    scatter_data()

if __name__ == "__main__":
    main()