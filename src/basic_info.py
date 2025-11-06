import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown


def get_basic_info(df: pd.DataFrame, remove_duplicates: bool = False) -> pd.DataFrame:
    display(Markdown("## 🧾 Basic DataFrame Overview"))
    
    # Head
    display(Markdown("### 🔹 Head of the DataFrame"))
    display(df.head())
    
    # Shape
    display(Markdown("### 📐 Shape of the DataFrame"))
    display(Markdown(f"- **Rows:** {df.shape[0]}  \n- **Columns:** {df.shape[1]}"))
    
    # Combined Table: Columns, Missing Values, Data Types, Unique Values
    display(Markdown("### 🧩 Column Summary"))
    summary_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.values,
        "Missing Values": df.isna().sum().values,
        "% Missing": (df.isna().sum() / len(df) * 100).round(2).values,
        "Unique Values": [df[col].nunique(dropna=True) for col in df.columns],
        "Sample Values": [
            ", ".join(map(str, df[col].dropna().unique()[:5])) 
            if df[col].nunique(dropna=True) <= 10 or df[col].dtype == "object" 
            else "Numerical/Continuous"
            for col in df.columns
        ]
    })
    
    display(
        summary_df.style
        .background_gradient(subset=["% Missing"], cmap="Reds")
        .set_caption("Column Summary: Types, Missingness, and Unique Values")
    )
    
    # Duplicates
    display(Markdown("### 📋 Duplicate Rows"))
    duplicates = df.duplicated().sum()
    display(Markdown(f"- Found **{duplicates}** duplicate rows"))
    
    if remove_duplicates and duplicates > 0:
        df = df.drop_duplicates()
        display(Markdown(f"✅ Removed **{duplicates}** duplicate rows. New shape: {df.shape}"))
    
    # Visualization of missing values (optional)
    if df.isna().sum().sum() > 0:
        display(Markdown("### 📊 Missing Values Heatmap"))
        plt.figure(figsize=(10, 4))
        sns.heatmap(df.isna(), cbar=False, cmap="Reds")
        plt.title("Missing Values Heatmap", fontsize=12)
        plt.show()
    
    display(Markdown("✅ **Data overview complete!**"))
    return df
