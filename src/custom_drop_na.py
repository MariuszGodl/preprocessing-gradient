import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def custom_drop_na(df, cols_to_drop : list = None,
                   drop_col : bool = False, drop_col_threshold : float = 0.5,
                   drop_row : bool = False, drop_row_threshold : float = 0.5
                   ) -> tuple[pd.DataFrame, list[str], pd.DataFrame]: 
                    # [ original df, columns_droped, rows_droped ] 
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("The provided DataFrame is empty.")
    
    if not isinstance(drop_col_threshold, float) or drop_col_threshold <= 0:
        raise ValueError("drop_col_threshold must be a positive float.")
    if not isinstance(drop_row_threshold, float) or drop_row_threshold <= 0:
        raise ValueError("drop_row_treshold must be a positive float.")
    
    if cols_to_drop is None and not drop_col and not drop_row:
        raise Exception("You know you didn't drop anything, specify the parameters")
    
    if cols_to_drop is None:
        cols_to_drop = []
    elif not isinstance(cols_to_drop, (list, tuple, set)):
        raise TypeError("cols_to_drop must be a list, tuple, or set.")
    
    nr_of_rows = df.shape[0]

    cols_dropped = []
    for col in df.columns:
        nr_of_na_in_col = df[col].isna().sum()
        col_ratio = nr_of_na_in_col / nr_of_rows
        if col_ratio >= drop_col_threshold:
            cols_dropped.append(col)
    df.drop(columns=cols_dropped, inplace=True)

    
    nr_of_columns = df.shape[1]

    def f1(row):
        row_na_count = row.isna().sum()
        row_ratio = row_na_count / nr_of_columns
        return row_ratio > drop_row_threshold

    mask = df.apply(f1, axis=1)

    dropped_rows = df[mask]
    df = df[~mask]
    
    return df, cols_dropped, dropped_rows