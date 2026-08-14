"""
Classifier Factory implementing the 17 Classical ML Models & Baselines (Section 3.3).
Supports GPU acceleration for XGBoost, LightGBM, CatBoost, cuML, and scikit-learn models.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier, SGDClassifier
)
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
)

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class XGBClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper around XGBClassifier that transparently handles string or non-integer target encoding
    to prevent XGBoost 1.6+ ValueError on string target classes.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.clf = XGBClassifier(**kwargs)
        self.le = None

    def get_params(self, deep=True):
        return self.clf.get_params(deep=deep)

    def set_params(self, **params):
        self.clf.set_params(**params)
        return self

    def fit(self, X, y, **fit_params):
        if y is not None:
            y_arr = np.asarray(y)
            if y_arr.dtype.kind in {'U', 'S', 'O'} or (len(y_arr) > 0 and isinstance(y_arr.flat[0], str)):
                self.le = LabelEncoder()
                y_encoded = self.le.fit_transform(y_arr)
                self.classes_ = self.le.classes_
                self.clf.fit(X, y_encoded, **fit_params)
                return self
            else:
                self.classes_ = np.unique(y_arr)
        self.clf.fit(X, y, **fit_params)
        return self

    def predict(self, X, **predict_params):
        preds = self.clf.predict(X, **predict_params)
        if self.le is not None:
            return self.le.inverse_transform(preds)
        return preds

    def predict_proba(self, X, **predict_params):
        return self.clf.predict_proba(X, **predict_params)

    @property
    def feature_importances_(self):
        return getattr(self.clf, "feature_importances_", None)


try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

# Dynamic cuML GPU imports
try:
    import cuml
    HAS_CUML = True
except ImportError:
    HAS_CUML = False

import scipy.sparse


class CumlSparseToDenseAdapter:
    """
    Adapter for cuML estimators that only accept dense float32 inputs (e.g. LinearSVC, SVC, RandomForest, KNeighbors).
    Converts sparse CSR matrix to dense np.float32 on the fly and ensures outputs are NumPy arrays.
    """
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        if scipy.sparse.issparse(X):
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
            X = X.toarray().astype(np.float32)
        elif isinstance(X, np.ndarray) and X.dtype != np.float32:
            X = X.astype(np.float32)
        preds = self.estimator.predict(X, **kwargs)
        if hasattr(preds, "to_numpy"):
            preds = preds.to_numpy()
        elif hasattr(preds, "get"):
            preds = preds.get()
        return np.asarray(preds)

    def predict_proba(self, X, **kwargs):
        if scipy.sparse.issparse(X):
            X = X.toarray().astype(np.float32)
        elif isinstance(X, np.ndarray) and X.dtype != np.float32:
            X = X.astype(np.float32)
        if hasattr(self.estimator, "predict_proba"):
            preds = self.estimator.predict_proba(X, **kwargs)
        elif hasattr(self.estimator, "decision_function"):
            df = self.estimator.decision_function(X)
            if hasattr(df, "to_numpy"):
                df = df.to_numpy()
            elif hasattr(df, "get"):
                df = df.get()
            df = np.asarray(df)
            prob = 1.0 / (1.0 + np.exp(-df))
            if prob.ndim == 1:
                return np.vstack([1 - prob, prob]).T
            return prob
        else:
            raise AttributeError(f"{self.estimator.__class__.__name__} has no predict_proba or decision_function")
        if hasattr(preds, "to_numpy"):
            preds = preds.to_numpy()
        elif hasattr(preds, "get"):
            preds = preds.get()
        return np.asarray(preds)

    def decision_function(self, X, **kwargs):
        if scipy.sparse.issparse(X):
            X = X.toarray().astype(np.float32)
        elif isinstance(X, np.ndarray) and X.dtype != np.float32:
            X = X.astype(np.float32)
        preds = self.estimator.decision_function(X, **kwargs)
        if hasattr(preds, "to_numpy"):
            preds = preds.to_numpy()
        elif hasattr(preds, "get"):
            preds = preds.get()
        return np.asarray(preds)

    def __getattr__(self, name):
        return getattr(self.estimator, name)


def get_classifier(model_name: str, use_gpu: bool = True, random_state: int = 42, **kwargs) -> Any:
    """
    Instantiates a classification model by name with GPU acceleration support (cuML/PyTorch/Native GPU).
    Falls back gracefully to CPU scikit-learn if cuML/GPU is unavailable.
    """
    m = model_name.strip()
    gpu_active = use_gpu and torch.cuda.is_available() and HAS_CUML
    
    if m == "DummyClassifier":
        p = {"strategy": "most_frequent", **kwargs}
        return DummyClassifier(**p)
        
    elif m == "LogisticRegression":
        if gpu_active:
            try:
                cp = {"max_iter": 1000, **{k: v for k, v in kwargs.items() if k not in ("random_state", "n_jobs")}}
                return CumlSparseToDenseAdapter(cuml.linear_model.LogisticRegression(**cp))
            except Exception:
                pass
        p = {"max_iter": 1000, "random_state": random_state, "n_jobs": -1, **kwargs}
        return LogisticRegression(**p)
        
    elif m == "LinearSVC":
        if gpu_active:
            try:
                cp = {"max_iter": 2000, **{k: v for k, v in kwargs.items() if k not in ("random_state", "dual")}}
                return CumlSparseToDenseAdapter(cuml.svm.LinearSVC(**cp))
            except Exception:
                pass
        p = {"random_state": random_state, "max_iter": 2000, "dual": "auto", **kwargs}
        return LinearSVC(**p)
        
    elif m == "SVC_linear":
        if gpu_active:
            try:
                cp = {"kernel": "linear", "probability": True, "max_iter": 5000, **{k: v for k, v in kwargs.items() if k not in ("random_state", "cache_size")}}
                return CumlSparseToDenseAdapter(cuml.svm.SVC(**cp))
            except Exception:
                pass
        p = {"kernel": "linear", "probability": True, "max_iter": 5000, "cache_size": 1000, "random_state": random_state, **kwargs}
        return SVC(**p)
        
    elif m == "SVC_rbf" or m == "SVC":
        if gpu_active:
            try:
                cp = {"kernel": "rbf", "probability": True, "max_iter": 5000, **{k: v for k, v in kwargs.items() if k not in ("random_state", "cache_size")}}
                return CumlSparseToDenseAdapter(cuml.svm.SVC(**cp))
            except Exception:
                pass
        p = {"kernel": "rbf", "probability": True, "max_iter": 5000, "cache_size": 1000, "random_state": random_state, **kwargs}
        return SVC(**p)
        
    elif m == "SVC_poly":
        if gpu_active:
            try:
                cp = {"kernel": "poly", "probability": True, "max_iter": 5000, **{k: v for k, v in kwargs.items() if k not in ("random_state", "cache_size")}}
                return CumlSparseToDenseAdapter(cuml.svm.SVC(**cp))
            except Exception:
                pass
        p = {"kernel": "poly", "probability": True, "max_iter": 5000, "cache_size": 1000, "random_state": random_state, **kwargs}
        return SVC(**p)
        
    elif m == "SVC_sigmoid":
        if gpu_active:
            try:
                cp = {"kernel": "sigmoid", "probability": True, "max_iter": 5000, **{k: v for k, v in kwargs.items() if k not in ("random_state", "cache_size")}}
                return CumlSparseToDenseAdapter(cuml.svm.SVC(**cp))
            except Exception:
                pass
        p = {"kernel": "sigmoid", "probability": True, "max_iter": 5000, "cache_size": 1000, "random_state": random_state, **kwargs}
        return SVC(**p)
        
    elif m == "RidgeClassifier":
        if gpu_active:
            try:
                cp = {k: v for k, v in kwargs.items() if k != "random_state"}
                return CumlSparseToDenseAdapter(cuml.linear_model.Ridge(**cp))
            except Exception:
                pass
        p = {"random_state": random_state, **kwargs}
        return RidgeClassifier(**p)
        
    elif m == "PassiveAggressiveClassifier":
        p = {"max_iter": 1000, "random_state": random_state, "n_jobs": -1, **kwargs}
        return PassiveAggressiveClassifier(**p)
        
    elif m == "SGDClassifier":
        if gpu_active:
            try:
                cp = {"epochs": 1000, **{k: v for k, v in kwargs.items() if k not in ("random_state", "n_jobs", "max_iter")}}
                return CumlSparseToDenseAdapter(cuml.linear_model.MBSGDClassifier(**cp))
            except Exception:
                pass
        p = {"max_iter": 1000, "random_state": random_state, "n_jobs": -1, **kwargs}
        return SGDClassifier(**p)
        
    elif m == "MultinomialNB":
        if gpu_active:
            try:
                return cuml.naive_bayes.MultinomialNB(**kwargs)
            except Exception:
                pass
        p = {**kwargs}
        return MultinomialNB(**p)
        
    elif m == "ComplementNB":
        if gpu_active:
            try:
                return cuml.naive_bayes.ComplementNB(**kwargs)
            except Exception:
                pass
        p = {**kwargs}
        return ComplementNB(**p)
        
    elif m == "KNeighborsClassifier":
        if gpu_active:
            try:
                cp = {k: v for k, v in kwargs.items() if k != "n_jobs"}
                return CumlSparseToDenseAdapter(cuml.neighbors.KNeighborsClassifier(**cp))
            except Exception:
                pass
        p = {"n_jobs": -1, **kwargs}
        return KNeighborsClassifier(**p)
        
    elif m == "MLPClassifier":
        p = {"hidden_layer_sizes": (100,), "max_iter": 300, "early_stopping": True, "n_iter_no_change": 10, "random_state": random_state, **kwargs}
        return MLPClassifier(**p)
        
    elif m == "DecisionTreeClassifier":
        p = {"random_state": random_state, **kwargs}
        return DecisionTreeClassifier(**p)
        
    elif m == "RandomForestClassifier":
        if gpu_active:
            try:
                cp = {"n_estimators": 100, "random_state": random_state, **{k: v for k, v in kwargs.items() if k != "n_jobs"}}
                return CumlSparseToDenseAdapter(cuml.ensemble.RandomForestClassifier(**cp))
            except Exception:
                pass
        p = {"n_estimators": 100, "random_state": random_state, "n_jobs": -1, **kwargs}
        return RandomForestClassifier(**p)
        
    elif m == "ExtraTreesClassifier":
        p = {"n_estimators": 100, "random_state": random_state, "n_jobs": -1, **kwargs}
        return ExtraTreesClassifier(**p)
        
    elif m == "AdaBoostClassifier":
        p = {"algorithm": "SAMME", "random_state": random_state, **kwargs}
        return AdaBoostClassifier(**p)
        
    elif m == "GradientBoostingClassifier":
        p = {"random_state": random_state, **kwargs}
        return GradientBoostingClassifier(**p)
        
    elif m == "XGBClassifier":
        if not HAS_XGB:
            raise ImportError("xgboost is not installed.")
        device_arg = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        p = {
            "device": device_arg, "tree_method": "hist", "random_state": random_state,
            "eval_metric": "logloss", **kwargs
        }
        return XGBClassifierWrapper(**p)
        
    elif m == "LGBMClassifier":
        if not HAS_LGBM:
            raise ImportError("lightgbm is not installed.")
        device_arg = "gpu" if (use_gpu and torch.cuda.is_available()) else "cpu"
        p = {"device": device_arg, "random_state": random_state, "verbose": -1, **kwargs}
        return LGBMClassifier(**p)
        
    elif m == "CatBoostClassifier":
        if not HAS_CAT:
            raise ImportError("catboost is not installed.")
        task_arg = "GPU" if (use_gpu and torch.cuda.is_available()) else "CPU"
        p = {"task_type": task_arg, "random_seed": random_state, "verbose": 0, **kwargs}
        return CatBoostClassifier(**p)
        
    else:
        raise ValueError(f"Unknown classifier model name: {model_name}")


ALL_CLASSIFIER_NAMES = [
    "DummyClassifier", "LogisticRegression", "LinearSVC", "SVC_linear",
    "SVC_rbf", "SVC_poly", "SVC_sigmoid", "RidgeClassifier",
    "PassiveAggressiveClassifier", "SGDClassifier", "MultinomialNB",
    "ComplementNB", "KNeighborsClassifier", "MLPClassifier",
    "DecisionTreeClassifier", "RandomForestClassifier", "ExtraTreesClassifier",
    "AdaBoostClassifier", "GradientBoostingClassifier",
    "XGBClassifier", "LGBMClassifier", "CatBoostClassifier"
]

