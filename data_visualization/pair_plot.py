import matplotlib
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
matplotlib.use('TkAgg')

def load(path: str) -> pd.DataFrame:
    """Read a CSV datasheet and return a DataFrame."""
    df = pd.read_csv(path)
    return df

def pair_data():    
    try:
        data = load("../data_sets/dataset_train.csv")
        seaborn.pairplot(data, hue ='Hogwarts House')
        plt.title('What features are you going to use for your logistic regression?')
        plt.show()
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    pair_data()

if __name__ == "__main__":
    main()