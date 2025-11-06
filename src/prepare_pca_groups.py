import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import AgglomerativeClustering

def prepare_pca_groups(df, 
                       variance_threshold=0.01,
                       corr_method='spearman',
                       corr_threshold=0.8,
                       scale=True,
                       return_group_dfs=True,
                       max_categories_for_label=10):
    """
    Automatically identify groups of correlated columns for PCA.
    Handles categorical and datetime features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset (numeric + non-numeric allowed).
    variance_threshold : float, optional
        Threshold for removing low-variance features.
    corr_method : {'pearson', 'spearman'}, optional
        Correlation method.
    corr_threshold : float, optional
        Threshold for grouping columns.
    scale : bool, optional
        Whether to standardize numeric features.
    return_group_dfs : bool, optional
        Return ready-to-use PCA DataFrames.
    max_categories_for_label : int, optional
        Categorical columns with <= this many unique values use LabelEncoder;
        otherwise, OneHotEncoder is used.

    Returns
    -------
    dict
        {
            'groups': dict of grouped columns,
            'reduced_data': dict of DataFrames (if return_group_dfs=True),
            'dropped_low_variance': list of dropped columns,
            'encoded_columns': list of encoded categorical columns
        }
    """

    df_proc = df.copy()
    encoded_columns = []

    # --- 1️⃣ Handle datetime columns
    for col in df_proc.select_dtypes(include=['datetime', 'datetimetz']).columns:
        df_proc[col + '_timestamp'] = df_proc[col].astype('int64') // 1e9
        df_proc.drop(columns=col, inplace=True)

    # --- 2️⃣ Handle categorical columns
    cat_cols = df_proc.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        n_unique = df_proc[col].nunique()
        if n_unique <= max_categories_for_label:
            # Low-cardinality: use label encoding
            le = LabelEncoder()
            df_proc[col] = le.fit_transform(df_proc[col].astype(str))
            encoded_columns.append(col)
        else:
            # High-cardinality: use one-hot encoding
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe_df = pd.DataFrame(ohe.fit_transform(df_proc[[col]]),
                                  columns=[f"{col}_{cat}" for cat in ohe.categories_[0]],
                                  index=df_proc.index)
            df_proc = pd.concat([df_proc.drop(columns=col), ohe_df], axis=1)
            encoded_columns.append(col)

    # --- 3️⃣ Keep only numeric columns for PCA
    df_num = df_proc.select_dtypes(include=[np.number]).copy()

    # --- 4️⃣ Remove low-variance columns
    vt = VarianceThreshold(threshold=variance_threshold)
    vt.fit(df_num)
    kept_cols = df_num.columns[vt.get_support()]
    dropped_cols = list(set(df_num.columns) - set(kept_cols))
    df_num = df_num[kept_cols]

    # --- 5️⃣ Compute correlation matrix
    corr = df_num.corr(method=corr_method).fillna(0).abs()

    # --- 6️⃣ Convert to distance matrix for clustering
    distance = 1 - corr
    np.fill_diagonal(distance.values, 0)

    # --- 7️⃣ Cluster correlated columns
    clustering = AgglomerativeClustering(
        affinity='precomputed',
        linkage='average',
        distance_threshold=1 - corr_threshold,
        n_clusters=None
    )
    clustering.fit(distance)

    # --- 8️⃣ Group columns by cluster
    groups = {}
    for cluster_id in np.unique(clustering.labels_):
        cols_in_group = df_num.columns[clustering.labels_ == cluster_id].tolist()
        groups[f"group_{cluster_id+1}"] = cols_in_group

    # --- 9️⃣ Optionally scale and output group-wise DataFrames
    reduced_data = {}
    if return_group_dfs:
        for group_name, cols in groups.items():
            data = df_num[cols]
            if scale:
                data = pd.DataFrame(StandardScaler().fit_transform(data), columns=cols)
            reduced_data[group_name] = data

    return {
        "groups": groups,
        "reduced_data": reduced_data if return_group_dfs else None,
        "dropped_low_variance": dropped_cols,
        "encoded_columns": encoded_columns
    }
