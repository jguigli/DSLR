import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# matplotlib.use('TkAgg')

def load(path: str) -> pd.DataFrame:
    """Read a CSV datasheet and return a DataFrame."""
    df = pd.read_csv(path)
    return df

def scatter_data():    
    try:
        data = load("../data_sets/dataset_train.csv")
        
        # Herbology / Defense Against the Dark Arts

        # data_y = data['Index']
        # plt.scatter(data_x, data_y, color = 'blue')
        
        data_x = data['Astronomy']
        data_y = data['Defense Against the Dark Arts']
        plt.scatter(data_x, data_y, color = 'green')

        plt.title('What are the two features that are similar ?')
        plt.ylabel('Index')
        plt.xlabel('Herbology / Defense Against the Dark Arts')
        plt.show()
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    scatter_data()

if __name__ == "__main__":
    main()