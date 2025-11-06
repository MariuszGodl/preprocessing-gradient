import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

def multiplot_norm(df : pd.DataFrame, hue : str, 
                   cols_to_exclude : list = None, 
                   nr_of_plots_col : int = 3, create_kde : bool = True):
    """
    Creates a grid of normalized histograms (optionally with KDEs) for all numeric 
    and categorical columns in a DataFrame, visualizing how their distributions 
    differ across a specified categorical variable (hue).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to plot.
    
    hue : str
        The name of the categorical column used for color grouping in the plots.
    
    cols_to_exclude : list, optional
        A list of column names to exclude from plotting. Default is None.
    
    nr_of_plots_col : int, optional
        Number of plots per row in the grid layout. Must be a positive integer. Default is 3.
    
    create_kde : bool, optional
        Whether to overlay Kernel Density Estimate (KDE) curves on numeric histograms. Default is True.

    Raises
    ------
    TypeError
        If input types are invalid (e.g., `df` not a DataFrame, `cols_to_exclude` not list-like).
    ValueError
        If DataFrame is empty, the hue column is missing, 
        the hue has too many unique categories, 
        or no valid columns remain for plotting.

    Notes
    -----
    - Columns with fewer than 10 unique values are treated as categorical.
    - Histograms are normalized to show probabilities (`stat='probability'`).
    - Each subplot title follows the format: "<column> vs <hue_column>".
    - Empty or unsupported columns are labeled with error messages in red.
    
    Returns
    -------
    None
        Displays a matplotlib figure with multiple subplots comparing distributions.
    """
    
    # --- VALIDATION ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("The DataFrame is empty.")
    if hue not in df.columns:
        raise ValueError(f"'{hue}' not found in DataFrame columns.")
    if not isinstance(nr_of_plots_col, int) or nr_of_plots_col <= 0:
        raise ValueError("nr_of_plots_col must be a positive integer.")
    
    if cols_to_exclude is None:
        cols_to_exclude = []
    elif not isinstance(cols_to_exclude, (list, tuple, set)):
        raise TypeError("cols_to_exclude must be a list, tuple, or set.")
    
    if df[hue].nunique() > 10:
        raise ValueError(f"'{hue}' has too many unique values ({df[hue].nunique()}). "
                         "Consider grouping or categorizing it.")
    
    # --- SETUP ---
    plot_columns = [col for col in df.columns if col not in cols_to_exclude and col != hue]
    if not plot_columns:
        raise ValueError("No valid columns to plot after exclusions.")
    
    nr_of_plots = len(plot_columns)
    nr_of_plots_row = math.ceil(nr_of_plots / nr_of_plots_col)

    fig, axes = plt.subplots(
        nr_of_plots_row,
        nr_of_plots_col,
        figsize=(5 * nr_of_plots_col, 4 * nr_of_plots_row)
    )
    axes = np.atleast_1d(axes).flatten()

    # --- PLOTTING ---
    for i, col in enumerate(plot_columns):
        ax = axes[i]
        try:
            # Skip columns with all NaN
            if df[col].dropna().empty:
                ax.text(0.5, 0.5, f"'{col}' is empty", ha='center', va='center', color='red')
                continue
            
            # Categorical or low-unique numeric
            if df[col].dtype == 'object' or df[col].nunique() < 10:
                sns.histplot(
                    data=df, x=col, hue=hue,
                    stat="probability", common_norm=True,
                    multiple='stack', ax=ax
                )
            
            # Numeric features
            elif pd.api.types.is_numeric_dtype(df[col]):
                sns.histplot(
                    data=df, x=col, hue=hue, bins=20, kde=create_kde,
                    stat="probability", common_norm=True,
                    multiple='stack', ax=ax
                )
            else:
                ax.text(0.5, 0.5, f"Unsupported dtype: {df[col].dtype}",
                        ha='center', va='center', color='red')
            
            ax.set_title(f"{col} vs {hue}")
            ax.tick_params(axis='x', rotation=25)
        
        except Exception as e:
            ax.text(0.5, 0.5, f"Error plotting '{col}': {e}", ha='center', va='center', color='red')
    
    # Remove unused axes
    for j in range(len(plot_columns), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
