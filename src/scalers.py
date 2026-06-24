import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        np_list = np.array(X, dtype=float)
        
        self.mean_ = np.mean(np_list, axis=0)
        self.std_ = np.std(np_list, axis=0)
        
        # Prevent division by zero for constant features
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        np_list = np.array(X, dtype=float)
        return (np_list - self.mean_) / self.std_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X, y=None):
        np_list = np.array(X, dtype=float)
        self.max_ = np.max(np_list, axis=0)
        self.min_ = np.min(np_list, axis=0)
        return self

    def transform(self, X):
        np_list = np.array(X, dtype=float)
        range_diff = self.max_ - self.min_
        
        # Avoid dividing by zero
        range_diff[range_diff == 0] = 1.0
        return (np_list - self.min_) / range_diff

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class RobustScaler:
    def __init__(self):
        self.median_ = None
        self.IQR_ = None

    def fit(self, X, y=None):
        np_list = np.array(X, dtype=float)
        
        self.median_ = np.median(np_list, axis=0)
        
        q25, q75 = np.percentile(np_list, [25, 75], axis=0)
        self.IQR_ = q75 - q25
        
        # Avoid division by zero
        self.IQR_[self.IQR_ == 0] = 1.0
        return self

    def transform(self, X):
        np_list = np.array(X, dtype=float)
        return (np_list - self.median_) / self.IQR_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)