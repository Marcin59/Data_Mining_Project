import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_data():
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv").merge(pd.read_csv("./data/sample_submission.csv"), how="inner")
    y_train = train["SalePrice"]
    X_train = train.drop(columns = ["SalePrice"])
    y_test = test["SalePrice"]
    X_test = test.drop(columns = ["SalePrice"])
    return X_train, y_train, X_test, y_test