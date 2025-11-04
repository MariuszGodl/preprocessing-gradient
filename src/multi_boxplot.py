import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
def multiboxplot(df, cols_to_exclude=None, nr_of_plots_col=3) -> list[list[str, pd.DataFrame]]:

    def get_outliers(column_of_interest, iqr_const=1.5) -> list[str, pd.DataFrame]:
        q1 = df[column_of_interest].quantile(0.25)
        q3 = df[column_of_interest].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_const * iqr
        upper_bound = q3 + iqr_const * iqr

        # Filter outliers
        outliers = df[(df[column_of_interest] < lower_bound) | (df[column_of_interest] > upper_bound)]

        # Return DataFrame of outliers for this column
        return [column_of_interest, outliers]

    if cols_to_exclude is None:
        cols_to_exclude=[]

    nr_of_plots = df.shape[1] - len(cols_to_exclude) 
    nr_of_plots_rows = math.ceil(nr_of_plots /nr_of_plots_col)

    fig, axes = plt.subplots(nr_of_plots, nr_of_plots_rows, figsize=[5*nr_of_plots_col, 10*nr_of_plots_rows])
    axes = axes.flatten()
    
    outliers = []
    i = 0 
    for col in df.columns:
        if col in cols_to_exclude:
            continue
        
        # we do not want objects and I want to skip categorical numerric varibles
        if df[col].dtype in ["object", "category"] or df[col].nunique() < 10:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            sns.boxplot(data=df, y=col, ax=axes[i])
            outliers.append(get_outliers(col))
            axes[i].set_title(f"{col}")
            i += 1 

    for j in range(i, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()

    return outliers