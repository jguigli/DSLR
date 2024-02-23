import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
matplotlib.use('TkAgg')
from logistic_regression.utils import load

def scatter_data():    
    try:
        data = load("../data_sets/dataset_train.csv")
        
        data_x = data['Astronomy']
        data_y = data['Defense Against the Dark Arts']
        plt.scatter(data_x, data_y, color = 'green')

        plt.title('What are the two features that are similar ?')
        plt.xlabel('Astronomy')
        plt.ylabel('Defense Against the Dark Arts')
        plt.show()
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    scatter_data()

if __name__ == "__main__":
    main()