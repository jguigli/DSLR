import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
matplotlib.use('TkAgg')
from logistic_regression.utils import load

def histogram_data():
    try:
        data = load("../data_sets/dataset_train.csv")
        
        # Arithmancy
        # or 
        # Care of Magical Creatures

        N = data['Arithmancy'].size
        bins = int(1 + math.log2(N))
        colors = ['red', 'green', 'blue', 'yellow']
        hogwart_house = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']

        for color, house in zip(colors, hogwart_house):
            data_course = data[data['Hogwarts House'] == house]['Arithmancy']
            plt.hist(data_course, bins, density=True, histtype='bar', color=color, label=house, alpha=0.7)

        plt.title('Which Hogwarts course has a homogeneous score distribution between all four houses?')
        plt.xlabel('Arithmancy')
        plt.ylabel('Density')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    histogram_data()

if __name__ == "__main__":
    main()