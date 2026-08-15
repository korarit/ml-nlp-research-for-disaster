"""
Regressor Factory implementing the 17 Continuous Count Regression Models (Task 2 Approach B2).
Supports GPU acceleration for XGBoost, LightGBM, CatBoost, cuML, and scikit-learn regressors.
"""

from typing import Any
import torch
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import (
    Ridge, PassiveAggressiveRegressor, SGDRegressor
)
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

# Dynamic cuML GPU imports
try:
    import cuml
    HAS_CUML = True
except ImportError:
    HAS_CUML = False


import numpy as np
import scipy.sparse


class CumlSparseToDenseAdapter:
    """
    Adapter for cuML estimators that only accept dense float32 inputs (e.g. LinearSVR, SVR, RandomForest, KNeighbors).
    Converts sparse CSR matrix to dense np.float32 on the fly with chunked batch inference to prevent OOM.
    """
    def __init__(self, estimator, max_dense_mb: float = 300.0, chunk_size: int = 10000):
        self.estimator = estimator
        self.max_dense_mb = max_dense_mb
        self.chunk_size = chunk_size

    def fit(self, X, y=None, **kwargs):
        if scipy.sparse.issparse(X):
            n_samples, n_features = X.shape
            dense_mb = (n_samples * n_features * 4) / (1024 * 1024)
            if dense_mb > self.max_dense_mb:
                raise MemoryError(
                    f"cuML dense fitting requires {dense_mb:.1f} MB RAM (exceeds {self.max_dense_mb} MB limit). "
                    "Falling back to sparse-native solver."
                )
            X = X.toarray().astype(np.float32)
        elif isinstance(X, np.ndarray) and X.dtype != np.float32:
            X = X.astype(np.float32)
        if y is not None:
            if hasattr(y, "values"):
                y = y.values
            if isinstance(y, np.ndarray) and y.dtype == np.float64:
                y = y.astype(np.float32)
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        if scipy.sparse.issparse(X):
            n_samples, n_features = X.shape
            dense_mb = (n_samples * n_features * 4) / (1024 * 1024)
            if dense_mb > 50.0:  # Chunk large sparse matrices
                all_preds = []
                for i in range(0, n_samples, self.chunk_size):
                    chunk = X[i : i + self.chunk_size].toarray().astype(np.float32)
                    p = self.estimator.predict(chunk, **kwargs)
                    if hasattr(p, "to_numpy"):
                        p = p.to_numpy()
                    elif hasattr(p, "get"):
                        p = p.get()
                    all_preds.append(np.asarray(p))
                return np.concatenate(all_preds)
            else:
                X = X.toarray().astype(np.float32)
        elif isinstance(X, np.ndarray) and X.dtype != np.float32:
            X = X.astype(np.float32)
        preds = self.estimator.predict(X, **kwargs)
        if hasattr(preds, "to_numpy"):
            preds = preds.to_numpy()
        elif hasattr(preds, "get"):
            preds = preds.get()
        return np.asarray(preds)

    def __getattr__(self, name):
        return getattr(self.estimator, name)


def get_regressor(model_name: str, use_gpu: bool = True, random_state: int = 42, **kwargs) -> Any:
    """
    Instantiates a regression model by name with optional GPU acceleration (cuML/PyTorch/Native GPU).
    Falls back gracefully to CPU scikit-learn if cuML/GPU is unavailable.
    """
    m = model_name.strip()
    gpu_active = use_gpu and torch.cuda.is_available() and HAS_CUML
    
    if m == "DummyRegressor":
        p = {"strategy": "mean", **kwargs}
        return DummyRegressor(**p)
        
    elif m == "Ridge":
        if gpu_active:
            try:
                cp = {k: v for k, v in kwargs.items() if k != "random_state"}
                return CumlSparseToDenseAdapter(cuml.linear_model.Ridge(**cp))
            except Exception:
                pass
        p = {"random_state": random_state, **kwargs}
        return Ridge(**p)
        
    elif m == "LinearSVR":
        if gpu_active:
            try:
                cp = {"max_iter": 2000, **{k: v for k, v in kwargs.items() if k not in ("random_state", "dual")}}
                return CumlSparseToDenseAdapter(cuml.svm.LinearSVR(**cp))
            except Exception:
                pass
        dual_val = kwargs.pop("dual", False)
        p = {"random_state": random_state, "max_iter": 2000, "dual": dual_val, **kwargs}
        return LinearSVR(**p)
        
    elif m == "SVR_linear":
        if gpu_active:
            try:
                cp = {"kernel": "linear", "max_iter": 5000, **{k: v for k, v in kwargs.items() if k not in ("random_state", "cache_size")}}
                return CumlSparseToDenseAdapter(cuml.svm.SVR(**cp))
            except Exception:
                pass
        p = {"kernel": "linear", "max_iter": 5000, "cache_size": 1000, **kwargs}
        return SVR(**p)
        
    elif m == "SVR_rbf" or m == "SVR":
        if gpu_active:
            try:
                cp = {"kernel": "rbf", "max_iter": 5000, **{k: v for k, v in kwargs.items() if k not in ("random_state", "cache_size")}}
                return CumlSparseToDenseAdapter(cuml.svm.SVR(**cp))
            except Exception:
                pass
        p = {"kernel": "rbf", "max_iter": 5000, "cache_size": 1000, **kwargs}
        return SVR(**p)
        
    elif m == "SVR_poly":
        if gpu_active:
            try:
                cp = {"kernel": "poly", "max_iter": 5000, **{k: v for k, v in kwargs.items() if k not in ("random_state", "cache_size")}}
                return CumlSparseToDenseAdapter(cuml.svm.SVR(**cp))
            except Exception:
                pass
        p = {"kernel": "poly", "max_iter": 5000, "cache_size": 1000, **kwargs}
        return SVR(**p)
        
    elif m == "SVR_sigmoid":
        if gpu_active:
            try:
                cp = {"kernel": "sigmoid", "max_iter": 5000, **{k: v for k, v in kwargs.items() if k not in ("random_state", "cache_size")}}
                return CumlSparseToDenseAdapter(cuml.svm.SVR(**cp))
            except Exception:
                pass
        p = {"kernel": "sigmoid", "max_iter": 5000, "cache_size": 1000, **kwargs}
        return SVR(**p)
        
    elif m == "PassiveAggressiveRegressor":
        p = {"max_iter": 1000, "random_state": random_state, **kwargs}
        return PassiveAggressiveRegressor(**p)
        
    elif m == "SGDRegressor":
        if gpu_active:
            try:
                cp = {"epochs": 1000, **{k: v for k, v in kwargs.items() if k not in ("random_state", "max_iter")}}
                return CumlSparseToDenseAdapter(cuml.linear_model.MBSGDRegressor(**cp))
            except Exception:
                pass
        p = {"max_iter": 1000, "random_state": random_state, **kwargs}
        return SGDRegressor(**p)
        
    elif m == "KNeighborsRegressor":
        if gpu_active:
            try:
                cp = {k: v for k, v in kwargs.items() if k != "n_jobs"}
                return CumlSparseToDenseAdapter(cuml.neighbors.KNeighborsRegressor(**cp))
            except Exception:
                pass
        p = {"n_jobs": -1, **kwargs}
        return KNeighborsRegressor(**p)
        
    elif m == "MLPRegressor":
        p = {"hidden_layer_sizes": (100,), "max_iter": 300, "early_stopping": True, "n_iter_no_change": 10, "random_state": random_state, **kwargs}
        return MLPRegressor(**p)
        
    elif m == "DecisionTreeRegressor":
        p = {"random_state": random_state, **kwargs}
        return DecisionTreeRegressor(**p)
        
    elif m == "RandomForestRegressor":
        if gpu_active:
            try:
                cp = {"n_estimators": 100, "random_state": random_state, **{k: v for k, v in kwargs.items() if k != "n_jobs"}}
                return CumlSparseToDenseAdapter(cuml.ensemble.RandomForestRegressor(**cp))
            except Exception:
                pass
        p = {"n_estimators": 100, "random_state": random_state, "n_jobs": -1, **kwargs}
        return RandomForestRegressor(**p)
        
    elif m == "ExtraTreesRegressor":
        p = {"n_estimators": 100, "random_state": random_state, "n_jobs": -1, **kwargs}
        return ExtraTreesRegressor(**p)
        
    elif m == "AdaBoostRegressor":
        base_tree = DecisionTreeRegressor(max_depth=3, max_features="sqrt", random_state=random_state)
        p = {"estimator": base_tree, "random_state": random_state, **kwargs}
        return AdaBoostRegressor(**p)
        
    elif m == "GradientBoostingRegressor":
        p = {"max_features": "sqrt", "random_state": random_state, **kwargs}
        return GradientBoostingRegressor(**p)
        
    elif m == "XGBRegressor":
        if not HAS_XGB:
            raise ImportError("xgboost is not installed.")
        device_arg = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        p = {"device": device_arg, "tree_method": "hist", "random_state": random_state, **kwargs}
        return XGBRegressor(**p)
        
    elif m == "LGBMRegressor":
        if not HAS_LGBM:
            raise ImportError("lightgbm is not installed.")
        device_arg = "gpu" if (use_gpu and torch.cuda.is_available()) else "cpu"
        p = {"device": device_arg, "random_state": random_state, "verbose": -1, **kwargs}
        return LGBMRegressor(**p)
        
    elif m == "CatBoostRegressor":
        if not HAS_CAT:
            raise ImportError("catboost is not installed.")
        task_arg = "GPU" if (use_gpu and torch.cuda.is_available()) else "CPU"
        p = {"task_type": task_arg, "random_seed": random_state, "verbose": 0, **kwargs}
        return CatBoostRegressor(**p)
        
    else:
        raise ValueError(f"Unknown regressor model name: {model_name}")


ALL_REGRESSOR_NAMES = [
    "DummyRegressor", "Ridge", "LinearSVR", "SVR_linear",
    "SVR_rbf", "SVR_poly", "SVR_sigmoid", "PassiveAggressiveRegressor",
    "SGDRegressor", "KNeighborsRegressor", "MLPRegressor",
    "DecisionTreeRegressor", "RandomForestRegressor", "ExtraTreesRegressor",
    "AdaBoostRegressor", "GradientBoostingRegressor",
    "XGBRegressor", "LGBMRegressor", "CatBoostRegressor"
]

