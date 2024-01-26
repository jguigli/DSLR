import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load(path: str) -> pd.DataFrame:
    """Read a CSV datasheet and return a DataFrame."""
    df = pd.read_csv(path)
    return df

def export_thetas(theta0, theta1):
    df = pd.DataFrame({"theta0": [theta0],"theta1": [theta1]})
    df.to_csv("../data/thetas.csv", index=False)
    print(f"Exporting file : theta0 and theta1 has been saved to /data/thetas.csv\n")

def min_max_scaling(data):
    scaled_data = (data / np.max(data))
    return scaled_data

def adjust_coefficients(theta0_normalized, theta1_normalized, price, mileage):
    min_vals = np.min(price)
    max_vals = np.max(price)
    theta0 = theta0_normalized * np.max(price)
    theta1 = theta1_normalized * (np.max(price) / np.max(mileage))
    return theta0, theta1

def estimate_price(theta0, theta1, mileage):
    return theta0 + theta1 * mileage

def gradient_descent(mileage: np.ndarray, price: np.ndarray):
    m = float(len(mileage))
    theta0, theta1 = 0, 0
    learning_rate = 0.1
    costs = []
    i = 0

    while 1:
        old_theta0 = theta0
        old_theta1 = theta1

        predicted_price = estimate_price(theta0, theta1, mileage)
        diff_price = predicted_price - price

        d_theta0 = (1 / m) * np.sum(diff_price)
        d_theta1 = (1 / m) * np.sum(diff_price * mileage)

        theta0 -= learning_rate * d_theta0
        theta1 -= learning_rate * d_theta1

        if (theta0 == old_theta0 and theta1 == old_theta1):
            print(f"The linear regression has finished.\nNumber of iterations : {i}\n")
            break
        i += 1

    return theta0, theta1


def training_model():
    try:
        data = load("../data/data.csv")

        data_mileage = data['km'].astype('int')
        data_price = data['price'].astype('int')
        mileage = np.array(data_mileage)
        price = np.array(data_price)
        mileage_normalized = min_max_scaling(mileage)
        price_normalized = min_max_scaling(price)

        theta0, theta1 = gradient_descent(mileage_normalized, price_normalized)
        theta0, theta1 = adjust_coefficients(theta0, theta1, price, mileage)
        print(f"- theta0 : {theta0}")
        print(f"- theta1 : {theta1}")
        export_thetas(theta0, theta1)
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    training_model()

if __name__ == "__main__":
    main()