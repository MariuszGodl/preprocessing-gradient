import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

def multiboxplot(df: pd.DataFrame, 
                 hue: str = None, 
                 cols_to_exclude: list[str] = None, 
                 nr_of_plots_col: int = 3, 
                 iqr_const: float = 1.5) -> list[list[str, pd.DataFrame]]:
    """
    Generate multiple boxplots for all numeric columns in a DataFrame (excluding specified ones),
    optionally grouped by a categorical variable, and return DataFrames containing outlier rows 
    for each numeric column based on the Interquartile Range (IQR) method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data to visualize.
    
    hue : str, optional
        Name of the categorical column used for color grouping in boxplots.
        If None, no grouping is applied. Default is None.
    
    cols_to_exclude : list[str], optional
        List of column names to exclude from visualization and outlier detection.
        Default is None.
    
    nr_of_plots_col : int, optional
        Number of boxplots per row in the figure layout.
        Must be a positive integer. Default is 3.
    
    iqr_const : float, optional
        Constant multiplier for the Interquartile Range (IQR) used to determine 
        outlier thresholds. Outliers are defined as values outside the range 
        [Q1 - iqr_const * IQR, Q3 + iqr_const * IQR]. Default is 1.5.

    Returns
    -------
    outliers : list[list[str, pd.DataFrame]]
        A list of pairs where each element contains:
        - str: the column name,
        - pd.DataFrame: a subset of the original DataFrame containing detected outlier rows.
        
        Example:
        [
            ["column_1", outlier_df_1],
            ["column_2", outlier_df_2],
            ...
        ]

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame or if `cols_to_exclude` is not list-like.
    
    ValueError
        If the DataFrame is empty or no numeric columns are available for plotting.

    Notes
    -----
    - Only numeric columns with at least 10 unique values are included in the plots.
    - Each boxplot visualizes the column distribution and potential outliers.
    - The function also identifies and returns DataFrames of outlier rows for each column.
    - Subplots are arranged in a grid determined by `nr_of_plots_col`.
    - Any unsupported or failed plots are annotated with error messages in red text.

    Returns
    -------
    None
        Displays a matplotlib figure containing all generated boxplots.
    """

    
    # --- VALIDATION ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("The provided DataFrame is empty.")
    if not isinstance(nr_of_plots_col, int) or nr_of_plots_col <= 0:
        raise ValueError("nr_of_plots_col must be a positive integer.")
    
    if cols_to_exclude is None:
        cols_to_exclude = []
    elif not isinstance(cols_to_exclude, (list, tuple, set)):
        raise TypeError("cols_to_exclude must be a list, tuple, or set.")

    # --- HELPER FUNCTION ---
    def get_outliers(column_name: str, iqr_const: float = 1.5) -> list:
        """Return column name and a DataFrame of its outliers based on IQR."""
        series = df[column_name].dropna()
        if series.empty:
            return [column_name, pd.DataFrame(columns=df.columns)]
        
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - iqr_const * iqr, q3 + iqr_const * iqr
        
        outlier_rows = df[(df[column_name] < lower) | (df[column_name] > upper)]
        return [column_name, outlier_rows]

    numeric_columns = [
        col for col in df.columns
        if col not in cols_to_exclude
        and pd.api.types.is_numeric_dtype(df[col])
        and df[col].nunique() >= 10  
    ]

    if not numeric_columns:
        raise ValueError("No numeric columns available for plotting.")

    nr_of_plots = len(numeric_columns)
    nr_of_plots_row = math.ceil(nr_of_plots / nr_of_plots_col)

    fig, axes = plt.subplots(
        nr_of_plots_row,
        nr_of_plots_col,
        figsize=(5 * nr_of_plots_col, 4 * nr_of_plots_row)
    )
    axes = np.atleast_1d(axes).flatten()

    outliers = []
    for i, col in enumerate(numeric_columns):
        ax = axes[i]
        try:
            sns.boxplot(data=df, y=col, ax=ax, hue=hue)
            ax.set_title(col)
            outliers.append(get_outliers(col, iqr_const))
        except Exception as e:
            ax.text(0.5, 0.5, f"Error plotting '{col}': {e}", ha="center", va="center", color="red")
    
    for j in range(len(numeric_columns), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
    
    return outliers
