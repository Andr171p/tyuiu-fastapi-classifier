import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer


class BinaryImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        self.columns = columns
        self.label_binarizers = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in self.columns:
            label_binarizer = LabelBinarizer()
            label_binarizer.fit(X[col].unique())
            self.label_binarizers[col] = label_binarizer
        return self

    def transform(self, X: pd.DataFrame):
        X_copy = X.copy()
        for col in self.columns:
            binarized_col = self.label_binarizers[col].transform(X_copy[col]).astype(int)
            X_copy[f"{col}"] = binarized_col
            # X_copy.drop(col, axis=1, inplace=True)
        return X_copy