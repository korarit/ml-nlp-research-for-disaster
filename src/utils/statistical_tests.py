"""
Statistical Significance Testing Suite for Model Comparison.
Implements McNemar's Test, Holm-Bonferroni Correction, and Wilcoxon Signed-Rank Test.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.stats import wilcoxon, chi2, binom


def run_mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray
) -> Dict[str, float]:
    """
    Runs McNemar's Test on predictions of Model A vs Model B against Ground Truth.
    Contingency table:
      b: Model A correct & B wrong
      c: Model B correct & A wrong
    """
    y_t = np.array(y_true, dtype=str)
    y_a = np.array(y_pred_a, dtype=str)
    y_b = np.array(y_pred_b, dtype=str)
    
    correct_a = (y_a == y_t)
    correct_b = (y_b == y_t)
    
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    
    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "table_b": 0, "table_c": 0}
        
    if b + c < 25:
        # Exact Binomial test
        k = min(b, c)
        p_val = min(1.0, 2.0 * float(binom.cdf(k, b + c, 0.5)))
        stat = float((abs(b - c) - 1) ** 2 / (b + c))
    else:
        # Chi-square with continuity correction
        stat = float((abs(b - c) - 1) ** 2 / (b + c))
        p_val = float(chi2.sf(stat, df=1))
        
    return {
        "statistic": stat,
        "p_value": p_val,
        "table_b": b,
        "table_c": c
    }


def apply_holm_bonferroni(p_values: List[float]) -> List[float]:
    """
    Applies Holm-Bonferroni Correction to a list of p-values to control FWER.
    Returns adjusted p-values p_adj.
    """
    m = len(p_values)
    if m == 0:
        return []
        
    # Sort indices
    sorted_indices = np.argsort(p_values)
    adjusted_p = [0.0] * m
    
    cum_max = 0.0
    for k, idx in enumerate(sorted_indices):
        p = p_values[idx]
        adj = p * (m - k)
        cum_max = max(cum_max, adj)
        adjusted_p[idx] = min(cum_max, 1.0)
        
    return adjusted_p


def run_wilcoxon_residual_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray
) -> Dict[str, float]:
    """
    Runs Wilcoxon Signed-Rank Test on sample-wise absolute error residuals (|y_true - y_pred|).
    Used specifically for Task 2 Count Regressors comparison.
    """
    err_a = np.abs(np.array(y_true, dtype=float) - np.array(y_pred_a, dtype=float))
    err_b = np.abs(np.array(y_true, dtype=float) - np.array(y_pred_b, dtype=float))
    
    diff = err_a - err_b
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0}
        
    try:
        res = wilcoxon(err_a, err_b, zero_method="wilcox")
        return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}
    except Exception:
        return {"statistic": 0.0, "p_value": 1.0}


def run_pairwise_model_stat_tests(
    y_true: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
    is_regression: bool = False
) -> List[Dict[str, Any]]:
    """
    Runs pairwise statistical comparison among models.
    Applies McNemar's test for classification or Wilcoxon for regression,
    followed by Holm-Bonferroni correction.
    """
    model_names = list(model_predictions.keys())
    pairs = []
    p_vals = []
    raw_results = []
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m_a = model_names[i]
            m_b = model_names[j]
            preds_a = model_predictions[m_a]
            preds_b = model_predictions[m_b]
            
            if is_regression:
                res = run_wilcoxon_residual_test(y_true, preds_a, preds_b)
                test_type = "Wilcoxon"
            else:
                res = run_mcnemar_test(y_true, preds_a, preds_b)
                test_type = "McNemar"
                
            p_vals.append(res["p_value"])
            raw_results.append({
                "model_a": m_a,
                "model_b": m_b,
                "test_type": test_type,
                "statistic": res["statistic"],
                "p_value": res["p_value"]
            })
            
    # Apply Holm-Bonferroni correction
    adj_p_vals = apply_holm_bonferroni(p_vals)
    
    for idx, item in enumerate(raw_results):
        item["p_adjusted"] = adj_p_vals[idx]
        item["significant"] = bool(adj_p_vals[idx] < 0.05)
        
    return raw_results
