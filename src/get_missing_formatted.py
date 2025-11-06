import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

def get_missing_formatted(
    df: pd.DataFrame, 
    only_missing: bool = False,
    fill: bool = False,
    fill_func: callable = None
) -> pd.DataFrame:
    """
    Returns a summary DataFrame showing % of missing values and number of unique values
    for each column in the input DataFrame. Optionally fills missing values based on
    group or custom logic.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    only_missing : bool, optional
        If True, returns only columns that have missing values. Default is False.
    fill : bool, optional
        If True, missing values are filled using default logic or a provided fill_func. Default is False.
    fill_func : scalar, optional  
        Custom value to fill missing entries. This value is passed directly to `fillna()`.  
        Example: fill_func=0 or fill_func=df["col"].mean().

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['Column', '% Missing', 'Missing Values Sum']
    """
    
    summary = pd.DataFrame({
        "Column": df.columns,
        "% Missing": (df.isna().sum() / len(df) * 100).round(2).values,
        "Missing Values Sum": df.isna().sum().values
    })
    
    if only_missing:
        summary = summary[summary["% Missing"] > 0]
    
    if fill:
        if fill_func is not None:
            df[col] = df.groupby("PriceCategory")[col].transform(lambda x: x.fillna(fill_func(x)))
        else:
            for col in summary["Column"]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df.groupby("PriceCategory")[col].transform(lambda x: x.fillna(x.mean()))
                else:
                    df[col] = df.groupby("PriceCategory")[col].transform(
                        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else "Unknown")
                    )
    
    return summary.reset_index(drop=True)
