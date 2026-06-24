import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, drop='first'):
        self.drop = drop
        self.categories_ = {}  # Maps: column_name -> list of kept categories
        self.col_names_ = []   # Final output column names

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.categories_ = {}
        self.col_names_ = []

        for col in X_df.columns:
            # Grab unique values, clean out NaNs, sort alphabetically
            unique_vals = sorted(list(X_df[col].dropna().unique()))
            
            # Prevent the dummy variable trap by dropping the first sorted category
            if self.drop == 'first' and len(unique_vals) > 1:
                kept_vals = unique_vals[1:] 
            else:
                kept_vals = unique_vals
                
            self.categories_[col] = kept_vals
            
            for val in kept_vals:
                self.col_names_.append(f"{col}_{val}")

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        out_dict = {}

        for col in X_df.columns:
            kept_vals = self.categories_[col]
            for val in kept_vals:
                # Generates a 1 or 0 integer column for every kept category
                out_dict[f"{col}_{val}"] = (X_df[col] == val).astype(int)

        res_df = pd.DataFrame(out_dict, index=X_df.index)
        
        if isinstance(X, pd.DataFrame):
            return res_df
        return res_df.values


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cv=5):
        self.cv = cv
        self.global_means_ = {}   # Memorized for the Test split
        self.overall_mean_ = 0.0  # Fallback safety net for totally unseen test categories

    def fit(self, X, y):
        """Memorizes the global category averages (Strictly used for unseen Test data)"""
        X_df = pd.DataFrame(X)
        y_arr = np.array(y)
        
        self.overall_mean_ = np.mean(y_arr)
        self.global_means_ = {}

        for col in X_df.columns:
            stats = pd.DataFrame({'cat': X_df[col], 'target': y_arr}).groupby('cat')['target'].mean()
            self.global_means_[col] = stats.to_dict()

        return self

    def transform(self, X):
        """Applies the memorized global means to the Test split."""
        X_df = pd.DataFrame(X).copy()
        out_df = pd.DataFrame(index=X_df.index)

        for col in X_df.columns:
            mapping = self.global_means_.get(col, {})
            # If the test set has a weird category not seen in training, drop the overall mean on it
            out_df[f"{col}_target_enc"] = X_df[col].map(mapping).fillna(self.overall_mean_)

        if isinstance(X, pd.DataFrame):
            return out_df
        return out_df.values

    def fit_transform(self, X, y):
        """
        The Leakage Defender: Computes out-of-fold target encoding for the Training set 
        so the regression model doesn't just memorize its own labels.
        """
        X_df = pd.DataFrame(X).copy()
        y_arr = np.array(y)
        
        # 1. Fit the normal global means first so .transform() works later on the test set
        self.fit(X_df, y_arr)

        out_df = pd.DataFrame(index=X_df.index)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for col in X_df.columns:
            oof_series = pd.Series(index=X_df.index, dtype=float)
            
            for train_idx, val_idx in kf.split(X_df):
                X_tr, y_tr = X_df.iloc[train_idx][col], y_arr[train_idx]
                X_val = X_df.iloc[val_idx][col]

                # Calculate target mean using ONLY the 4 training folds
                fold_means = pd.DataFrame({'cat': X_tr, 'target': y_tr}).groupby('cat')['target'].mean().to_dict()
                
                # Map those averages onto the 1 held-out fold
                oof_series.iloc[val_idx] = X_val.map(fold_means).fillna(self.overall_mean_)

            out_df[f"{col}_target_enc"] = oof_series

        if isinstance(X, pd.DataFrame):
            return out_df
        return out_df.values