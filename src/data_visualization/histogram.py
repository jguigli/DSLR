import sys
import math
import pandas as pd
from ..libs import CSV
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


def create_plot_hist_spread(
    df: pd.DataFrame,
    ax: any,
    column: str,
    title="Histogram",
    x_label="Value",
    y_label="Frequency",
):
    N = df.size
    bins = int(1 + math.log2(N))
    colors = ["red", "yellow", "blue", "green"]
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    for color, house in zip(colors, houses):
        df_course = df[df["Hogwarts House"] == house][column]
        ax.hist(
            df_course,
            bins=bins,
            alpha=0.8,
            label=house,
            color=color,
            density=True,
        )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def histogram_spread(numeric_df: pd.DataFrame):
    grid_y = 5
    grid_x = 3
    # grid_x = math.ceil(numeric_df.columns.size - 1 / 2 / grid_y)
    fig, ax = plt.subplots(grid_x, grid_y, figsize=(20, 8))
    ax = ax.flatten()
    for i in range(1, numeric_df.columns.size):
        column = numeric_df.columns[i]
        create_plot_hist_spread(
            numeric_df,
            ax[i - 1],
            column,
            title=column,
            x_label=column,
            y_label="Density",
        )
    for i in range(numeric_df.columns.size - 1, grid_x * grid_y):
        ax[i].axis("off")
    fig.subplots_adjust(hspace=0.3, wspace=0.3, bottom=0.05, top=0.95, left=0.05, right=0.95)
    plt.show()


def create_plot_hist_chain(
    df: pd.DataFrame,
    column: str,
    title="Histogram",
    x_label="Value",
    y_label="Frequency",
):
    N = df.size
    bins = int(1 + math.log2(N))
    colors = ["red", "yellow", "blue", "green"]
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    for color, house in zip(colors, houses):
        df_course = df[df["Hogwarts House"] == house][column]
        plt.hist(
            df_course,
            bins=bins,
            alpha=0.8,
            label=house,
            color=color,
            density=True,
        )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def histogram_chain(numeric_df: pd.DataFrame):
    for i in range(1, numeric_df.columns.size):
        column = numeric_df.columns[i]
        create_plot_hist_chain(
            numeric_df,
            column,
            title="score distribution accross all houses",
            x_label=column,
            y_label="Density",
        )


def histogram(path: str):
    try:
        df = CSV.load(path)
        house = df["Hogwarts House"].dropna()
        if house is None or house.size == 0:
            raise Exception("No house data found.")
        numeric_df = df.select_dtypes(include="number").dropna()
        if numeric_df is None or numeric_df.size == 0:
            raise Exception("No numerical data found.")
        numeric_df.insert(0, "Hogwarts House", house)
        histogram_chain(numeric_df)
        histogram_spread(numeric_df)
        column = "Arithmancy"
        create_plot_hist_chain(
            numeric_df,
            column,
            title="score distribution accross all houses",
            x_label=column,
            y_label="Density",
        )
    except Exception as e:
        print(f"Error: {e}")
        return


def main(*args):
    if args is None or len(args) != 2:
        print("Usage: python -m histogram <dataset>")
        return
    histogram(args[1])


if __name__ == "__main__":
    main(*sys.argv[:])
