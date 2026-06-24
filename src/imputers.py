import numpy as np
import pandas as pd

class SimpleImputer:
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X, y=None):
        # 1. Convert to a standard NumPy array WITHOUT forcing float yet
        X_arr = np.array(X)

        if self.strategy in ['mean', 'median']:
            # Only force float here, because mean/median require math
            try:
                X_num = X_arr.astype(float)
            except ValueError:
                raise ValueError(f"Cannot calculate {self.strategy} on non-numeric data containing strings.")

            if self.strategy == 'mean':
                self.statistics_ = np.nanmean(X_num, axis=0)
            else:
                self.statistics_ = np.nanmedian(X_num, axis=0)

        elif self.strategy in ['mode', 'most_frequent']:
            modes = []
            for col_idx in range(X_arr.shape[1]):
                col = X_arr[:, col_idx]
                
                # CRITICAL: pd.isna() safely checks for NaNs inside string arrays
                clean_mask = ~pd.isna(col)
                col_clean = col[clean_mask]

                if len(col_clean) == 0:
                    modes.append(np.nan)
                else:
                    vals, counts = np.unique(col_clean, return_counts=True)
                    modes.append(vals[np.argmax(counts)])

            # Save as object dtype so the array can store 'S', 'C', or numbers
            self.statistics_ = np.array(modes, dtype=object)

        else:
            raise ValueError(f"Unrecognized strategy: {self.strategy}")

        return self

    def transform(self, X):
        # 1. Convert to 'object' array so it can safely house both text and decimals
        X_arr = np.array(X, dtype=object).copy()

        for col_idx in range(X_arr.shape[1]):
            # Again, use pd.isna() to locate the missing slots safely
            nan_mask = pd.isna(X_arr[:, col_idx])
            X_arr[nan_mask, col_idx] = self.statistics_[col_idx]

        # 2. Cast back to float for numeric strategies so the output dtype
        #    matches sklearn's SimpleImputer (float64, not object)
        if self.strategy in ['mean', 'median']:
            X_arr = X_arr.astype(float)

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_arr, columns=X.columns, index=X.index)

        return X_arr

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    
class KNNImputer:
    def __init__(self, k=5):
        self.k = k
        self.train_X_ = None

    def fit(self, X, y=None):
        """Memorizes the reference training set."""
        self.train_X_ = np.array(X, dtype=float).copy()
        return self

    def transform(self, X):
        X_arr = np.array(X, dtype=float).copy()
        n_rows, n_cols = X_arr.shape

        for i in range(n_rows):
            for j in range(n_cols):
                if np.isnan(X_arr[i, j]):
                    
                    # 1. Identify which columns in Row [i] actually have numbers to measure distance with
                    valid_cols = [c for c in range(n_cols) if c != j and not np.isnan(X_arr[i, c])]

                    # Edge Case A: The row is completely empty. Fall back to the training column mean.
                    if len(valid_cols) == 0:
                        X_arr[i, j] = np.nanmean(self.train_X_[:, j])
                        continue

                    # 2. Filter the reference pool (self.train_X_). 
                    # A reference row is only a valid candidate neighbor IF:
                    #   a) It actually has a known value in column [j] (the answer we are looking for)
                    #   b) It has zero NaNs across the 'valid_cols' we are using to calculate distance
                    candidate_mask = ~np.isnan(self.train_X_[:, j])
                    for c in valid_cols:
                        candidate_mask &= ~np.isnan(self.train_X_[:, c])

                    candidates = self.train_X_[candidate_mask]

                    # Edge Case B: No reference rows survived the filter. Fall back to training mean.
                    if len(candidates) == 0:
                        X_arr[i, j] = np.nanmean(self.train_X_[:, j])
                        continue

                    # 3. Calculate Euclidean distance strictly across the shared valid_cols
                    target_point = X_arr[i, valid_cols]
                    candidate_points = candidates[:, valid_cols]
                    
                    diffs = candidate_points - target_point
                    distances = np.sqrt(np.sum(diffs ** 2, axis=1))

                    # 4. Grab the K nearest neighbors (or however many survived, if < K)
                    effective_k = min(self.k, len(distances))
                    nearest_indices = np.argsort(distances)[:effective_k]

                    # Grab their values from column [j] and take the average
                    X_arr[i, j] = np.mean(candidates[nearest_indices, j])

        # Preserve DataFrame structure if the user passed one in
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_arr, columns=X.columns, index=X.index)

        return X_arr

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)