"""
Metrics Utility Suite for Classification, Clinical Triage, NER, and Count Regression.
Calculates F1, F2, MCC, QWK, Under-Triage Rate with 95% CI (bootstrapping), MAE, RMSE, EM, Jaccard, Levenshtein.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score, log_loss,
    mean_absolute_error, root_mean_squared_error
)


def compute_f2_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary") -> float:
    """Computes F2-Score giving double weight to Recall: F2 = (5 * P * R) / (4 * P + R)."""
    return float(fbeta_score(y_true, y_pred, beta=2.0, average=average, zero_division=0))


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Computes full classification metric suite for Task 1 and Triage tasks."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    
    # Handle possible continuous / float predictions or 2D probability outputs
    if y_p.ndim > 1 and y_p.shape[1] > 1:
        y_p = np.argmax(y_p, axis=1)
    elif y_p.dtype.kind == 'f':
        y_p = np.round(y_p).astype(int)
    y_t = y_t.astype(int) if np.issubdtype(y_t.dtype, np.number) else y_t
    y_p = y_p.astype(int) if np.issubdtype(y_p.dtype, np.number) else y_p
    
    unique_all = np.union1d(np.unique(y_t), np.unique(y_p))
    is_binary = len(unique_all) <= 2 and set(unique_all).issubset({0, 1, 0.0, 1.0, "0", "1"})
    avg_mode = "binary" if is_binary else "weighted"
    
    acc = accuracy_score(y_t, y_p)
    prec = precision_score(y_t, y_p, average=avg_mode, zero_division=0)
    rec = recall_score(y_t, y_p, average=avg_mode, zero_division=0)
    f1_macro = f1_score(y_t, y_p, average="macro", zero_division=0)
    f1_weighted = f1_score(y_t, y_p, average="weighted", zero_division=0)
    f1_val = f1_score(y_t, y_p, average=avg_mode, zero_division=0)
    f2_val = compute_f2_score(y_t, y_p, average=avg_mode)
    
    mcc = matthews_corrcoef(y_t, y_p) if is_binary else 0.0
    kappa = cohen_kappa_score(y_t, y_p)
    
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1_val),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f2": float(f2_val),
        "mcc": float(mcc),
        "cohen_kappa": float(kappa)
    }
    
    if y_prob is not None:
        try:
            if is_binary and y_prob.ndim == 1:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
                metrics["log_loss"] = float(log_loss(y_true, y_prob))
            elif y_prob.ndim == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
                metrics["log_loss"] = float(log_loss(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.0
            metrics["log_loss"] = 0.0
            
    return metrics


def compute_triage_clinical_metrics(
    y_true: List[str],
    y_pred: List[str],
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes Triage safety metrics according to IITT standards:
    - RED (Critical), YELLOW (Urgent), GREEN (Non-urgent)
    - Under-Triage Rate (UR): Predicting GREEN/YELLOW when True is RED, or GREEN when True is YELLOW.
    - Critical Under-Triage Rate: Predicting GREEN/YELLOW when True is RED (Dangerous!).
    - Over-Triage Rate (OR): Predicting higher severity than True.
    - Quadratic Weighted Kappa (QWK)
    - 95% Confidence Interval for Under-Triage Rate via Bootstrapping.
    """
    labels = ["GREEN", "YELLOW", "RED"]
    label2idx = {l: i for i, l in enumerate(labels)}
    
    y_t_idx = np.array([label2idx.get(str(y).upper(), 0) for y in y_true])
    y_p_idx = np.array([label2idx.get(str(y).upper(), 0) for y in y_pred])
    
    qwk = cohen_kappa_score(y_t_idx, y_p_idx, weights="quadratic")
    f1_weighted = f1_score(y_t_idx, y_p_idx, average="weighted", zero_division=0)
    f2_weighted = compute_f2_score(y_t_idx, y_p_idx, average="weighted")
    acc = accuracy_score(y_t_idx, y_p_idx)
    
    # Under-triage: true severity > predicted severity
    under_mask = y_t_idx > y_p_idx
    under_rate = np.mean(under_mask) if len(under_mask) > 0 else 0.0
    
    # Critical Under-triage: True is RED (idx 2) but predicted <= 1 (YELLOW or GREEN)
    red_indices = np.where(y_t_idx == 2)[0]
    if len(red_indices) > 0:
        crit_under = np.mean(y_p_idx[red_indices] < 2)
    else:
        crit_under = 0.0
        
    # Over-triage: predicted severity > true severity
    over_mask = y_p_idx > y_t_idx
    over_rate = np.mean(over_mask) if len(over_mask) > 0 else 0.0
    
    # 95% CI Bootstrapping for Under-Triage Rate
    np.random.seed(seed)
    n_samples = len(y_t_idx)
    boot_rates = []
    if n_samples > 0:
        for _ in range(n_bootstraps):
            bs_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            if len(red_indices) > 0:
                bs_reds = np.where(y_t_idx[bs_indices] == 2)[0]
                if len(bs_reds) > 0:
                    r = np.mean(y_p_idx[bs_indices][bs_reds] < 2)
                else:
                    r = np.mean(y_t_idx[bs_indices] > y_p_idx[bs_indices])
            else:
                r = np.mean(y_t_idx[bs_indices] > y_p_idx[bs_indices])
            boot_rates.append(r)
            
        ci_lower = float(np.percentile(boot_rates, 2.5))
        ci_upper = float(np.percentile(boot_rates, 97.5))
    else:
        ci_lower, ci_upper = 0.0, 0.0
        
    return {
        "qwk": float(qwk),
        "triage_accuracy": float(acc),
        "f1_weighted": float(f1_weighted),
        "f2_weighted": float(f2_weighted),
        "under_triage_rate": float(under_rate),
        "critical_under_triage_rate": float(crit_under),
        "over_triage_rate": float(over_rate),
        "ur_ci_lower_95": ci_lower,
        "ur_ci_upper_95": ci_upper,
        "ur_ci_str": f"{crit_under * 100:.2f}% [{ci_lower * 100:.2f}% - {ci_upper * 100:.2f}%]"
    }


def compute_count_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes MAE, RMSE, and Exact Match for count regression models,
    handling GT zero/null instances separately to prevent zero-padding score inflation.
    """
    y_t = np.array(y_true, dtype=float)
    y_p = np.clip(np.array(y_pred, dtype=float), 0, None)  # count cannot be negative
    
    mae = mean_absolute_error(y_t, y_p)
    rmse = root_mean_squared_error(y_t, y_p)
    
    # Exact Match for integer counts
    em = np.mean(np.round(y_t) == np.round(y_p))
    
    # Non-zero GT metrics
    nonzero_mask = y_t > 0
    if np.sum(nonzero_mask) > 0:
        nonzero_mae = float(mean_absolute_error(y_t[nonzero_mask], y_p[nonzero_mask]))
        nonzero_rmse = float(root_mean_squared_error(y_t[nonzero_mask], y_p[nonzero_mask]))
        nonzero_em = float(np.mean(np.round(y_t[nonzero_mask]) == np.round(y_p[nonzero_mask])))
    else:
        nonzero_mae, nonzero_rmse, nonzero_em = 0.0, 0.0, 1.0
        
    # Zero GT over-prediction rate
    zero_mask = y_t == 0
    if np.sum(zero_mask) > 0:
        zero_overpred_rate = float(np.mean(np.round(y_p[zero_mask]) > 0))
    else:
        zero_overpred_rate = 0.0
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "exact_match": float(em),
        "nonzero_mae": nonzero_mae,
        "nonzero_rmse": nonzero_rmse,
        "nonzero_exact_match": nonzero_em,
        "zero_overpred_rate": zero_overpred_rate
    }


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes raw Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
        
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


import re


def sanitize_string_val(val: Any) -> str:
    """Helper to clean string values, turning NaN / None / 'nan' / '0.0,0.0' into empty string and normalizing comma whitespace."""
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip()
    if s.lower() in ("nan", "none", "null", "0.0,0.0", "0.0, 0.0"):
        return ""
    if "," in s:
        s = re.sub(r"^(?:พิกัด|ละติจูด|ลองจิจูด|ตำแหน่ง|coords?|lat/lng)[:\s]*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*,\s*", ",", s)
    return s


def compute_string_match_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    Computes Exact Match (EM), Precision, Recall, F1, Non-Null GT Match Rate, Null FP Rate,
    Jaccard Similarity, and Normalized Levenshtein for entity extractions.
    
    Ground Truth Policy:
    - GT is null AND Pred is null: True Negative (TN) -> Excluded from match rate to prevent score inflation.
    - GT is null BUT Pred is non-null: False Positive (FP) -> Penalized as over-extraction.
    - GT is non-null AND Pred matches: True Positive (TP).
    - GT is non-null BUT Pred is wrong/null: False Negative (FN).
    """
    tp = 0
    fp = 0
    fn = 0
    fp_null = 0
    
    gt_present_count = 0
    gt_null_count = 0
    
    jaccard_scores = []
    lev_distances = []
    
    for t_str, p_str in zip(y_true, y_pred):
        t = sanitize_string_val(t_str)
        p = sanitize_string_val(p_str)
        
        has_gt = len(t) > 0
        has_pred = len(p) > 0
        
        if has_gt:
            gt_present_count += 1
            if t.lower() == p.lower():
                tp += 1
            else:
                fn += 1
                if has_pred:
                    fp += 1  # wrong value predicted
        else:
            gt_null_count += 1
            if has_pred:
                fp += 1
                fp_null += 1
                
        # Jaccard set match
        set_t = set(t.split())
        set_p = set(p.split())
        union = set_t.union(set_p)
        if len(union) == 0:
            jaccard = 1.0 if len(set_t) == len(set_p) else 0.0
        else:
            jaccard = len(set_t.intersection(set_p)) / len(union)
        jaccard_scores.append(jaccard)
        
        # Normalized Levenshtein
        dist = levenshtein_distance(t, p)
        max_len = max(len(t), len(p))
        norm_dist = dist / max_len if max_len > 0 else 0.0
        lev_distances.append(norm_dist)
        
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    gt_match_rate = tp / gt_present_count if gt_present_count > 0 else 1.0
    null_fp_rate = fp_null / gt_null_count if gt_null_count > 0 else 0.0
    
    # Legacy Exact Match Rate over whole dataset (including TNs) for backward compatibility
    em_all = (tp + (gt_null_count - fp_null)) / max(len(y_true), 1)
    
    return {
        "exact_match_rate": float(em_all),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "gt_match_rate": float(gt_match_rate),
        "null_fp_rate": float(null_fp_rate),
        "gt_present_count": int(gt_present_count),
        "jaccard_similarity": float(np.mean(jaccard_scores)),
        "normalized_levenshtein": float(np.mean(lev_distances))
    }
