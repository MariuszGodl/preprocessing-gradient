import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

def multiplot_norm(df, column_to_hue, cols_to_exclude=None, nr_of_plots_col=3):
    if cols_to_exclude is None:
        cols_to_exclude = []
    
    nr_of_plots = df.shape[1] - len(cols_to_exclude) - 1
    nr_of_plots_row = math.ceil(nr_of_plots / nr_of_plots_col)

    fig, axes = plt.subplots(nr_of_plots_row, nr_of_plots_col, figsize=(5*nr_of_plots_col, 4*nr_of_plots_row))
    axes = axes.flatten()

    i = 0
    for col in df.columns:
        if col == column_to_hue or col in cols_to_exclude:
            continue

        # Categorical features (object dtype or few unique values)
        if df[col].dtype == 'object' or df[col].nunique() < 10:
            sns.histplot(data=df, x=col, hue=column_to_hue, bins=20, ax=axes[i],
                common_norm=False, stat="probability"
            )



        # Numeric features
        elif pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(
                data=df, x=col, hue=column_to_hue, bins=20, kde=True, ax=axes[i],
                common_norm=False, stat="probability"
            )

            #sns.histplot(data=df, x=col, hue=column_to_hue, bins=20, kde=True, ax=axes[i])

        axes[i].set_title(f"{col} vs {column_to_hue}")
        i += 1

    # Remove unused subplots
    for j in range(i, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()