import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        np_list = np.array(X, dtype=float)
        
        self.mean = np.mean(np_list, axis=0)
        self.std = np.std(np_list, axis=0)
        
        # Prevent division by zero for constant features
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        np_list = np.array(X, dtype=float)
        return (np_list - self.mean) / self.std

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X, y=None):
        np_list = np.array(X, dtype=float)
        self.max = np.max(np_list, axis=0)
        self.min = np.min(np_list, axis=0)
        return self

    def transform(self, X):
        np_list = np.array(X, dtype=float)
        range_diff = self.max - self.min
        
        # Avoid dividing by zero
        range_diff[range_diff == 0] = 1.0
        return (np_list - self.min) / range_diff

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class RobustScaler:
    def __init__(self):
        self.median = None
        self.IQR = None

    def fit(self, X, y=None):
        np_list = np.array(X, dtype=float)
        
        self.median = np.median(np_list, axis=0)
        
        q25, q75 = np.percentile(np_list, [25, 75], axis=0)
        self.IQR = q75 - q25
        
        # Avoid division by zero
        self.IQR[self.IQR == 0] = 1.0
        return self

    def transform(self, X):
        np_list = np.array(X, dtype=float)
        return (np_list - self.median) / self.IQR

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)