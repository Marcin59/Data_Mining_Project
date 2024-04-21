import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import skew

def load_data():
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv").merge(pd.read_csv("./data/sample_submission.csv"), how="inner")
    y_train = train["SalePrice"]
    X_train = train.drop(columns = ["SalePrice"])
    y_test = test["SalePrice"]
    X_test = test.drop(columns = ["SalePrice"])
    return X_train, y_train, X_test, y_test

def remove_null_columns(df: pd.DataFrame, threshold = 0.4):
    num_of_nulls = df.isna().sum()
    percentage_of_nulls = num_of_nulls/df.shape[0]
    null_columns = percentage_of_nulls.loc[percentage_of_nulls > threshold].index
    return df.drop(null_columns, axis = 1), null_columns

def remove_low_variance_columns(df: pd.DataFrame, threshold = 0.02):
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(df)
    removed_columns = df.columns[np.bitwise_not(sel.get_support())]
    return pd.DataFrame(sel.transform(df), columns = sel.get_feature_names_out()), removed_columns

def remove_high_correlation_columns(df: pd.DataFrame, threshold = 0.9):
    correlations = df.corr().iloc
    column_indexes_to_remove = set()
    for row_index, row in enumerate(correlations):
        if row_index in column_indexes_to_remove: continue
        for col_index, elem in enumerate(row[row_index+1:], start=row_index+1):
            if abs(elem) >  threshold:
                column_indexes_to_remove.add(col_index)
    columns_to_remove = df.columns[list(column_indexes_to_remove)]
    return df.drop(columns=columns_to_remove), columns_to_remove

def remove_low_correlation_to_target_columns(df: pd.DataFrame, target: np.ndarray, threshold = 0.05):
    df["target"] = target
    target_col_corr = df.corr()["target"]
    df = df.drop(["target"], axis=1)
    column_indexes_to_remove = set()
    for index, value in enumerate(target_col_corr.iloc):
        if abs(value) < threshold:
            column_indexes_to_remove.add(index)
    columns_to_remove = df.columns[list(column_indexes_to_remove)]
    return df.drop(columns=columns_to_remove), columns_to_remove

def test_model(model, X_train, y_train, X_test, y_test):
    errors = []
    for _ in range(50):
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        errors.append(mean_squared_error(y_test, prediction))
    return errors

def compare_errors(errors):
    items = list(errors.items())
    items.sort(key=lambda x: x[0])
    values = [k[1] for k in items]
    keys = [k[0] for k in items]
    fig = plt.figure(figsize =(10, 7))
    plt.violinplot(values)
    plt.xticks(range(1, len(keys) + 1), keys)
    plt.show()

class ColumnDropperTransformer():
    def __init__(self, threshold = 0.9):
        self.columns_to_remove = set(["Id", 'GarageArea', "GarageYrBlt", 'TotRmsAbvGrd', "1stFlrSF"])
        self.threshold = threshold

    def transform(self,X,y=None):
        return X.drop(columns = self.columns_to_remove)

    def fit(self, X: pd.DataFrame, y=None):
        num_of_nulls = X.isna().sum()
        percentage_of_nulls = num_of_nulls/X.shape[0]
        self.columns_to_remove.update(percentage_of_nulls[percentage_of_nulls > self.threshold].index)
        return self 
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in X_copy.columns:
            X_copy[col] = LabelEncoder().fit_transform(X_copy[col])
        return X_copy

class OutlierTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower=0.001, q_upper=0.999):
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.lower_bound = None
        self.upper_bound = None

    def fit(self, X, y=None):
        self.lower_bound = X.quantile(self.q_lower)
        self.upper_bound = X.quantile(self.q_upper)
        return self

    def transform(self, X: pd.DataFrame):
        for column in X.columns:
            lower_mask = X.loc[:, column] < self.lower_bound[column]
            upper_mask = X.loc[:, column] > self.upper_bound[column]
            X.loc[lower_mask, column] = self.lower_bound[column]
            X.loc[upper_mask, column] = self.upper_bound[column]
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

class SkewnessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.columns_to_transform = None

    def fit(self, X, y=None):
        skewness = X.apply(lambda x: skew(x.dropna()))
        self.columns_to_transform = skewness[skewness > self.threshold].index
        return self

    def transform(self, X):
        for col in self.columns_to_transform:
            X[col] = np.log1p(X[col])
        return X