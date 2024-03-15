import pandas as pd


class CSV:
    """CSV class to handle csv files"""

    @staticmethod
    def load(file_path, sep=",", header=0, index_col=0) -> pd.DataFrame:
        return pd.read_csv(file_path, sep=sep, header=header, index_col=index_col)

    @staticmethod
    def save(
        df: pd.DataFrame,
        file_path,
        file_name,
        sep=",",
        index=False,
        index_label="Index",
    ):
        try:
            df.to_csv(
                file_path + file_name, sep=sep, index=index, index_label=index_label
            )
        except Exception as e:
            print(f"Error: {e}")
