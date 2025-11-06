import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

def multiplot(df: pd.DataFrame, hue: str, 
              cols_to_exclude: list[str] = None, 
              nr_of_plots_col: int = 3, create_kde: bool = True) -> None:
    """
    Generate a grid of subplots visualizing the distributions of all columns in a DataFrame, 
    grouped by a specified categorical variable (hue). Automatically detects numerical 
    and categorical features and applies suitable plot types.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the dataset to visualize.
    
    hue : str
        Name of the categorical column used for color grouping in the plots.
        The column should contain a limited number of unique categories (<= 10).
    
    cols_to_exclude : list[str], optional
        List of column names to exclude from visualization. 
        Defaults to None (no exclusions).
    
    nr_of_plots_col : int, optional
        Number of subplots per row in the figure layout. 
        Must be a positive integer. Default is 3.
    
    create_kde : bool, optional
        If True, overlays Kernel Density Estimate (KDE) curves on numeric histograms.
        Default is True.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame, or `cols_to_exclude` is not a list-like object.
    
    ValueError
        If the DataFrame is empty, the hue column is missing, 
        has too many unique values (>10), 
        or no valid columns remain after exclusions.
    
    Notes
    -----
    - Columns with fewer than 10 unique values or of type `object` are treated as categorical.
    - Numeric columns are visualized using histograms (with optional KDE curves).
    - Categorical columns are visualized using count plots.
    - Each subplot title follows the format: "<column> vs <hue_column>".
    - Empty, invalid, or unsupported columns are labeled with error messages inside the plot.

    Returns
    -------
    None
        Displays a matplotlib figure with multiple subplots showing feature distributions.
    """

    
    # --- VALIDATION ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    
    if hue not in df.columns:
        raise ValueError(f"Column '{hue}' not found in DataFrame.")
    
    if df.empty:
        raise ValueError("The provided DataFrame is empty.")
    
    if cols_to_exclude is None:
        cols_to_exclude = []
    elif not isinstance(cols_to_exclude, (list, tuple, set)):
        raise TypeError("cols_to_exclude must be a list, tuple, or set.")
    
    if not isinstance(nr_of_plots_col, int) or nr_of_plots_col <= 0:
        raise ValueError("nr_of_plots_col must be a positive integer.")
    
    if df[hue].nunique() > 10:
        raise ValueError(f"Column '{hue}' has too many unique values. "
                        f"Consider categorizing it (unique values: {df[hue].nunique()}).")
    
    # --- PLOT SETUP ---
    # Columns to plot
    plot_columns = [col for col in df.columns if col not in cols_to_exclude and col != hue]
    
    if not plot_columns:
        raise ValueError("No valid columns left to plot after exclusions.")
    
    nr_of_plots = len(plot_columns)
    nr_of_plots_row = math.ceil(nr_of_plots / nr_of_plots_col)

    fig, axes = plt.subplots(
        nr_of_plots_row,
        nr_of_plots_col,
        figsize=(5 * nr_of_plots_col, 4 * nr_of_plots_row)
    )
    
    # Ensure axes is iterable
    if nr_of_plots_row * nr_of_plots_col == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # --- PLOTTING ---
    for i, col in enumerate(plot_columns):
        ax = axes[i]
        try:
            # Categorical features
            if df[col].dtype == 'object' or df[col].nunique() < 10:
                sns.countplot(data=df, x=col, hue=hue, ax=ax)
            
            # Numeric features
            elif pd.api.types.is_numeric_dtype(df[col]):
                sns.histplot(data=df, x=col, hue=hue, bins=20, kde=create_kde, ax=ax, multiple="stack")
            
            else:
                ax.text(0.5, 0.5, f"Unsupported dtype: {df[col].dtype}",
                        ha='center', va='center', fontsize=10, color='red')
            
            ax.set_title(f"{col} vs {hue}")
            ax.tick_params(axis='x', rotation=30)
        
        except Exception as e:
            ax.text(0.5, 0.5, f"Error plotting '{col}':\n{e}",
                    ha='center', va='center', fontsize=9, color='red')
    
    # Remove unused subplots
    for j in range(len(plot_columns), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
