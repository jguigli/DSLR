from ..libs import CSV
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys

matplotlib.use("TkAgg")


def create_scatter(
    x, y, title="Default", x_label="Values", y_label="Values", color="blue"
):
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, color=color, alpha=0.5, s=100, edgecolors="w")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def create_scatter_loop(
    df: pd.DataFrame,
    comp: str,
    offset=0,
    title="Scatter Plot",
):
    for course in df.columns[offset:]:
        print(f"Creating scatter plot for {comp} / {course}")
        create_scatter(
            df[comp].values,
            df[course].values,
            title=title,
            x_label=comp,
            y_label=course,
        )


def scatter_plot(path: str):
    try:
        df = CSV.load(path)
        numeric_df = df.select_dtypes(include="number")
        for column in numeric_df.columns:
            col = numeric_df[column].dropna()
            if col.size == 0:
                numeric_df = numeric_df.drop(column, axis=1)
        if numeric_df.size == 0:
            raise Exception("No numerical data found.")
        numeric_df = numeric_df.dropna()
        for i in range(0, numeric_df.columns.size):
            column = numeric_df.columns[i]
            create_scatter_loop(numeric_df, column, offset=i + 1)
        print(f"Creating scatter plot for Astronomy / Defense Against the Dark Arts")
        create_scatter(
            numeric_df["Astronomy"].values,
            numeric_df["Defense Against the Dark Arts"].values,
            title="Response to which courses are the most similar?",
            x_label="Astronomy",
            y_label="Defense Against the Dark Arts",
            color="green",
        )
        print(f"Creating scatter plot for Astronomy / Astronomy")
        create_scatter(
            numeric_df["Astronomy"].values,
            numeric_df["Astronomy"].values,
            title="Same Data",
            x_label="Astronomy",
            y_label="Astronomy",
            color="red",
        )
    except Exception as e:
        print(f"Error: {e}")
        return


def main(*args):
    if args is None or len(args) != 2:
        print("Usage: python -m scatter_plot <dataset>")
        return
    scatter_plot(args[1])


if __name__ == "__main__":
    main(*sys.argv)
