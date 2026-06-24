import itertools
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PolynomialFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, degree=2, include_bias=True, interaction_only=False):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only

    def fit(self, X, y=None):
        self.n_features_in_ = np.shape(X)[1]
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        n_samples, n_features = X_arr.shape

        output_matrices = []

        # 0-Degree term (Bias)
        if self.include_bias:
            output_matrices.append(np.ones((n_samples, 1)))

        # 1st-Degree terms (Raw features)
        if self.degree >= 1:
            output_matrices.append(X_arr)

        # 2nd-Degree terms
        if self.degree >= 2:
            if self.interaction_only:
                combos = itertools.combinations(range(n_features), 2)
            else:
                combos = itertools.combinations_with_replacement(
                    range(n_features), 2
                )

            # Vectorized pairwise column multiplication
            deg2_cols = [
                (X_arr[:, i] * X_arr[:, j]).reshape(-1, 1) for i, j in combos
            ]

            if deg2_cols:
                output_matrices.append(np.hstack(deg2_cols))

        return np.hstack(output_matrices)


class KBinsDiscretizer(BaseEstimator, TransformerMixin):

    def __init__(self, n_bins=10, strategy="uniform", encode="onehot-dense"):
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.n_features_in_ = X_arr.shape[1]
        self.bin_edges_ = []

        for col_idx in range(self.n_features_in_):
            col = X_arr[:, col_idx]

            if self.strategy == "uniform":
                edges = np.linspace(col.min(), col.max(), self.n_bins + 1)
            elif self.strategy == "quantile":
                edges = np.percentile(col, np.linspace(0, 100, self.n_bins + 1))
            else:
                raise ValueError(f"Unknown strategy: '{self.strategy}'")

            # Pad outer boundaries to Infinity to prevent test-set out-of-bounds errors
            edges[0] = -np.inf
            edges[-1] = np.inf
            self.bin_edges_.append(edges)

        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        encoded_columns = []

        for col_idx in range(self.n_features_in_):
            col = X_arr[:, col_idx]
            edges = self.bin_edges_[col_idx]

            # np.digitize returns 1-indexed bins; shift down to 0-indexed
            bin_indices = np.digitize(col, edges) - 1
            bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

            if self.encode == "onehot-dense":
                # Instantaneous one-hot mapping via Identity matrix indexing
                one_hot = np.eye(self.n_bins)[bin_indices]
                encoded_columns.append(one_hot)
            elif self.encode == "ordinal":
                encoded_columns.append(bin_indices.reshape(-1, 1))

        return np.hstack(encoded_columns)