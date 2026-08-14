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
                return cuml.linear_model.Ridge(**cp)
            except Exception:
                pass
        p = {"random_state": random_state, **kwargs}
        return Ridge(**p)
        
    elif m == "LinearSVR":
        if gpu_active:
            try:
                cp = {"max_iter": 2000, **{k: v for k, v in kwargs.items() if k not in ("random_state", "dual")}}
                return cuml.svm.LinearSVR(**cp)
            except Exception:
                pass
        p = {"random_state": random_state, "max_iter": 2000, "dual": "auto", **kwargs}
        return LinearSVR(**p)
        
    elif m == "SVR_linear":
        if gpu_active:
            try:
                cp = {"kernel": "linear", "max_iter": 5000, **{k: v for k, v in kwargs.items() if k not in ("random_state", "cache_size")}}
                return cuml.svm.SVR(**cp)
            except Exception:
                pass
        p = {"kernel": "linear", "max_iter": 5000, "cache_size": 1000, **kwargs}
        return SVR(**p)
        
    elif m == "SVR_rbf" or m == "SVR":
        if gpu_active:
            try:
                cp = {"kernel": "rbf", "max_iter": 5000, **{k: v for k, v in kwargs.items() if k not in ("random_state", "cache_size")}}
                return cuml.svm.SVR(**cp)
            except Exception:
                pass
        p = {"kernel": "rbf", "max_iter": 5000, "cache_size": 1000, **kwargs}
        return SVR(**p)
        
    elif m == "SVR_poly":
        if gpu_active:
            try:
                cp = {"kernel": "poly", "max_iter": 5000, **{k: v for k, v in kwargs.items() if k not in ("random_state", "cache_size")}}
                return cuml.svm.SVR(**cp)
            except Exception:
                pass
        p = {"kernel": "poly", "max_iter": 5000, "cache_size": 1000, **kwargs}
        return SVR(**p)
        
    elif m == "SVR_sigmoid":
        if gpu_active:
            try:
                cp = {"kernel": "sigmoid", "max_iter": 5000, **{k: v for k, v in kwargs.items() if k not in ("random_state", "cache_size")}}
                return cuml.svm.SVR(**cp)
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
                return cuml.linear_model.MBSGDRegressor(**cp)
            except Exception:
                pass
        p = {"max_iter": 1000, "random_state": random_state, **kwargs}
        return SGDRegressor(**p)
        
    elif m == "KNeighborsRegressor":
        if gpu_active:
            try:
                cp = {k: v for k, v in kwargs.items() if k != "n_jobs"}
                return cuml.neighbors.KNeighborsRegressor(**cp)
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
                return cuml.ensemble.RandomForestRegressor(**cp)
            except Exception:
                pass
        p = {"n_estimators": 100, "random_state": random_state, "n_jobs": -1, **kwargs}
        return RandomForestRegressor(**p)
        
    elif m == "ExtraTreesRegressor":
        p = {"n_estimators": 100, "random_state": random_state, "n_jobs": -1, **kwargs}
        return ExtraTreesRegressor(**p)
        
    elif m == "AdaBoostRegressor":
        p = {"random_state": random_state, **kwargs}
        return AdaBoostRegressor(**p)
        
    elif m == "GradientBoostingRegressor":
        p = {"random_state": random_state, **kwargs}
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

