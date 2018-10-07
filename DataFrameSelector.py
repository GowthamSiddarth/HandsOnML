from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes_names):
        self.attributes_names = attributes_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attributes_names].values
