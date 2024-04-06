from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class ColumnDropperTransformer():
    def __init__(self):
        self.columns_to_remove = set(["Id"])
        self.X: pd.DataFrame

    def transform(self,X,y=None):
        return X.drop(self.columns_to_remove,axis=1)

    def fit(self, X: pd.DataFrame, y=None):
        self.X = X
        self.columns_to_remove.update(self.get_low_variance_columns())
        self.columns_to_remove.update(self.get_null_columns())
        print(self.columns_to_remove)
        return self 
    
    def get_low_variance_columns(self, threshold = 0.015):
        min_max = MinMaxScaler()
        numerical_features = self.X.select_dtypes(exclude=object)
        normalized_numerical_features = min_max.fit_transform(numerical_features)
        selector = VarianceThreshold(threshold=threshold)
        selector.fit_transform(normalized_numerical_features)
        return numerical_features.columns[selector.get_support()]

    def get_null_columns(self, threshold = 0.4):
        num_of_nulls = self.X.isna().sum()
        percentage_of_nulls = num_of_nulls/self.X.shape[0]
        return percentage_of_nulls.loc[percentage_of_nulls > threshold].index