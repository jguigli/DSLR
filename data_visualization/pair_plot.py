import matplotlib
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
matplotlib.use('TkAgg')


def pair_data():
    try:
        data = pd.read_csv("../data_sets/dataset_train.csv")
        data = data.drop('Index', axis=1)
        seaborn.pairplot(data, hue ='Hogwarts House')
        # plt.title('What features are you going to use for your logistic regression?')
        plt.show()
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    pair_data()

if __name__ == "__main__":
    main()