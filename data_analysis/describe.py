import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load(path: str) -> pd.DataFrame:
    """Read a CSV datasheet and return a DataFrame."""
    df = pd.read_csv(path)
    return df

def describe():    
    try:
        data = load("../data/data.csv")
        data_mileage = data['km'].astype('int')
        data_price = data['price'].astype('int')
        mileage = np.array(data_mileage)
        price = np.array(data_price)
		
        thetas = load("../data/thetas.csv")
        theta0 = float(thetas['theta0'].iloc[0])
        theta1 = float(thetas['theta1'].iloc[0])

        predicted_price = estimate_price(theta0, theta1, mileage)

        plt.scatter(mileage, price)
        plt.plot([min(mileage), max(mileage)], [max(predicted_price), min(predicted_price)], color='red')
        plt.title('Linear regression')
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.show()
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    describe()

if __name__ == "__main__":
    main()