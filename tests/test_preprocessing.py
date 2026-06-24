import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.imputers import SimpleImputer, KNNImputer
from src.encoders import OneHotEncoder, TargetEncoder
from src.scalers import StandardScaler, MinMaxScaler, RobustScaler
from src.feature_creation import PolynomialFeatures, KBinsDiscretizer


# ─── SimpleImputer ──────────────────────────────────────────────────────────

def test_simple_imputer_mean_no_leakage():
    """fit() on train only; transform() on test must use train statistics."""
    X_train = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    X_test  = np.array([[np.nan, 2.0], [100.0, np.nan]])
    imp = SimpleImputer(strategy='mean')
    imp.fit(X_train)
    result = imp.transform(X_test)
    # Train mean of col 0 = (1+3+5)/3 = 3.0; col 1 = (4+6)/2 = 5.0
    assert result[0, 0] == pytest.approx(3.0)
    assert result[1, 1] == pytest.approx(5.0)


def test_simple_imputer_median():
    X = np.array([[1.0], [2.0], [np.nan], [100.0]])
    imp = SimpleImputer(strategy='median')
    result = imp.fit_transform(X)
    assert result[2, 0] == pytest.approx(2.0)  # median of [1,2,100] = 2.0


def test_simple_imputer_mode_categorical():
    X = pd.DataFrame({'embarked': ['S', 'S', 'C', None]})
    imp = SimpleImputer(strategy='mode')
    result = imp.fit_transform(X)
    assert result['embarked'].iloc[3] == 'S'


def test_simple_imputer_output_dtype_numeric():
    """Mean/median strategies must return float array, not object."""
    X = np.array([[1.0, np.nan], [3.0, 4.0]])
    imp = SimpleImputer(strategy='mean')
    result = imp.fit_transform(X)
    assert result.dtype == np.float64


def test_simple_imputer_matches_sklearn():
    from sklearn.impute import SimpleImputer as SkImp
    np.random.seed(42)
    X = np.random.randn(50, 4)
    X[::5, 0] = np.nan
    X[::7, 2] = np.nan
    for strategy in ['mean', 'median']:
        c = SimpleImputer(strategy=strategy).fit_transform(X)
        s = SkImp(strategy=strategy).fit_transform(X)
        np.testing.assert_array_almost_equal(c, s)


# ─── KNNImputer ─────────────────────────────────────────────────────────────

def test_knn_imputer_basic():
    X = np.array([
        [1.0, 2.0, 3.0],
        [1.1, 2.1, np.nan],
        [1.0, 2.0, 3.2],
    ])
    imp = KNNImputer(k=2)
    result = imp.fit_transform(X)
    assert not np.isnan(result[1, 2])
    assert 3.0 <= result[1, 2] <= 3.2


def test_knn_imputer_no_leakage():
    """fit() must only store training data; transform() must not modify training set."""
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_test  = np.array([[np.nan, 2.0]])
    imp = KNNImputer(k=1)
    imp.fit(X_train)
    result = imp.transform(X_test)
    assert not np.isnan(result).any()
    # Training set must be unchanged
    np.testing.assert_array_equal(imp.train_X_, X_train.astype(float))


# ─── OneHotEncoder ───────────────────────────────────────────────────────────

def test_ohe_drop_first():
    X = pd.DataFrame({'sex': ['male', 'female', 'male']})
    enc = OneHotEncoder(drop='first')
    out = enc.fit_transform(X)
    assert out.shape[1] == 1  # 2 categories - 1 dropped = 1 column


def test_ohe_unseen_category():
    """Unseen category in test set should produce all zeros."""
    X_train = pd.DataFrame({'embarked': ['S', 'C', 'Q']})
    X_test  = pd.DataFrame({'embarked': ['S', 'UNKNOWN']})
    enc = OneHotEncoder(drop='first')
    enc.fit(X_train)
    out = enc.transform(X_test)
    assert out.iloc[1].sum() == 0  # 'UNKNOWN' maps to all zeros


def test_ohe_pipeline_compatible():
    """OneHotEncoder must work after a SimpleImputer in a pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.base import BaseEstimator, TransformerMixin

    class SkImp(SimpleImputer, BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            super().fit(X, y)
            return self

    X = pd.DataFrame({'embarked': ['S', None, 'C', 'S']})
    pipe = Pipeline([('imp', SkImp(strategy='mode')), ('enc', OneHotEncoder(drop='first'))])
    result = pipe.fit_transform(X)
    assert not np.isnan(result.astype(float)).any().any()


# ─── TargetEncoder ───────────────────────────────────────────────────────────

def test_target_encoder_no_leakage():
    """fit_transform must not use the val fold's labels when encoding that fold."""
    X = pd.DataFrame({'cat': ['A'] * 50 + ['B'] * 50})
    y = np.array([1] * 50 + [0] * 50)  # A=always 1, B=always 0
    enc = TargetEncoder(cv=5)
    out = enc.fit_transform(X, y)
    # A rows should encode close to 1.0; B rows close to 0.0
    a_vals = out[X['cat'] == 'A']['cat_target_enc'].values
    b_vals = out[X['cat'] == 'B']['cat_target_enc'].values
    assert a_vals.mean() > 0.8
    assert b_vals.mean() < 0.2


# ─── Scalers ─────────────────────────────────────────────────────────────────

def test_standard_scaler_matches_sklearn():
    from sklearn.preprocessing import StandardScaler as Sk
    np.random.seed(0)
    X = np.random.randn(30, 3)
    c = StandardScaler().fit_transform(X)
    s = Sk().fit_transform(X)
    np.testing.assert_array_almost_equal(c, s)


def test_minmax_scaler_range():
    X = np.array([[1.0, 10.0], [5.0, 20.0], [3.0, 30.0]])
    result = MinMaxScaler().fit_transform(X)
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)


def test_robust_scaler_insensitive_to_outliers():
    X_clean = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    X_outlier = np.array([[1.0], [2.0], [3.0], [4.0], [1000.0]])
    sc_clean = RobustScaler().fit(X_clean)
    sc_outlier = RobustScaler().fit(X_outlier)
    # Median should be same; IQR should be very similar
    assert abs(sc_clean.median_[0] - sc_outlier.median_[0]) < 0.5


# ─── PolynomialFeatures ───────────────────────────────────────────────────────

def test_polynomial_features_output():
    X = np.array([[3.0, 5.0]])
    out = PolynomialFeatures(degree=2, include_bias=True).fit_transform(X)
    expected = np.array([[1.0, 3.0, 5.0, 9.0, 15.0, 25.0]])
    np.testing.assert_array_almost_equal(out, expected)


def test_polynomial_features_matches_sklearn():
    from sklearn.preprocessing import PolynomialFeatures as Sk
    np.random.seed(0)
    X = np.random.randn(10, 3)
    c = PolynomialFeatures(degree=2, include_bias=True).fit_transform(X)
    s = Sk(degree=2, include_bias=True).fit_transform(X)
    np.testing.assert_array_almost_equal(c, s)


# ─── KBinsDiscretizer ─────────────────────────────────────────────────────────

def test_kbins_onehot_shape():
    X = np.arange(100).reshape(-1, 1).astype(float)
    disc = KBinsDiscretizer(n_bins=10, strategy='uniform', encode='onehot-dense')
    out = disc.fit_transform(X)
    assert out.shape == (100, 10)


def test_kbins_quantile_balanced():
    """Quantile bins should have approximately equal counts."""
    X = np.random.randn(1000, 1)
    disc = KBinsDiscretizer(n_bins=10, strategy='quantile', encode='onehot-dense')
    out = disc.fit_transform(X)
    counts = out.sum(axis=0)
    assert counts.min() >= 80  # each bin should have ~100 obs