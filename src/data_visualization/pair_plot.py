from ..libs import CSV
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys

matplotlib.use("TkAgg")


def pair_plot(path: str):
    try:
        df = CSV.load(path)
        if (df is None) or (df.size == 0):
            raise Exception("No data found.")
        if df["Hogwarts House"].dropna().size == 0:
            raise Exception("No house data found.")
        sns.pairplot(df, hue="Hogwarts House")
        plt.show()
    except Exception as e:
        print(f"Error: {e}")
        return


def main(*args):
    if args is None or len(args) != 2:
        print("Usage: python -m pair_plot <dataset>")
        return
    pair_plot(args[1])


if __name__ == "__main__":
    main(*sys.argv[:])
