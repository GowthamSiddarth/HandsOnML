from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

rooms_idx, bedrooms_idx, population_idx, households_idx = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_idx] / X[:, households_idx]
        population_per_household = X[:, population_idx] / X[: households_idx]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[: bedrooms_idx] / X[: rooms_idx]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
