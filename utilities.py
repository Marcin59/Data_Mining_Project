import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def getNumericalColumns(df: pd.DataFrame):
    return df.select_dtypes(include=[np.int64, np.float64])

def fillNans(df: pd.DataFrame):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)