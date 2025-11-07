import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

def prepare_pca_groups(df : pd.DataFrame, 
                       col_of_intrest : str,
                       col_to_label_with_label_encoder : list[str] = None,
                       col_to_label_with_one_hot_encoding : list[str] = None,
                       col_to_label_with_ordinal_encoder : list[str] = None,
                       variance_threshold :float = 0.01) -> tuple[pd.DataFrame, dict]:

    col_to_label_with_label_encoder = col_to_label_with_label_encoder or []
    col_to_label_with_one_hot_encoding = col_to_label_with_one_hot_encoding or []
    col_to_label_with_ordinal_encoder = col_to_label_with_ordinal_encoder or []

    classes_created = {} 

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            # encode the value with one of two mtehods
            if col in col_to_label_with_one_hot_encoding:
                df = pd.get_dummies(df[col])
                    
                
            elif col in col_to_label_with_label_encoder:
                try:
                    pass
                except Exception as e:
                    raise e

            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                classes_created[col] = le.classes_
    return 

