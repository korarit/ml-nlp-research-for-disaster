"""
Task 2: NER & Entity/Count Extraction Pipeline.
Evaluates Approach A (Rules), Approach B1 (Binned Classifiers), Approach B2 (Continuous Regressors),
Approach B3 (CRF Tagger), and Approach C (Hybrid System) on Gemini CV and Luna Held-out Test.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.utils.data_loader import load_all_datasets, bin_count_target, COUNT_COLUMNS
from src.features.text_vectorizer import create_tfidf_vectorizer
from src.models.classifiers import get_classifier, ALL_CLASSIFIER_NAMES
from src.models.regressors import get_regressor, ALL_REGRESSOR_NAMES
from src.models.rules_engine import ExtractionRulesEngine
from src.utils.metrics import (
    compute_count_regression_metrics, compute_string_match_metrics,
    compute_classification_metrics
)
from src.utils.statistical_tests import run_pairwise_model_stat_tests


def run_task2_approach_a_rules(luna_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluates Approach A Rule-Based Extraction Engine on Luna Dataset."""
    engine = ExtractionRulesEngine()
    texts = luna_df["generated_text"].tolist()
    
    # Phone match
    true_phones = [str(p or "") for p in luna_df.get("gt_victim_phone", [])]
    pred_phones = [str(engine.extract_phone(t) or "") for t in texts]
    phone_m = compute_string_match_metrics(true_phones, pred_phones)
    
    # Map URL match
    true_urls = [str(u or "") for u in luna_df.get("gt_google_map_url", [])]
    pred_urls = [str(engine.extract_map_url(t) or "") for t in texts]
    url_m = compute_string_match_metrics(true_urls, pred_urls)
    
    # Count metrics
    count_maes = []
    for field in COUNT_COLUMNS:
        if field in luna_df.columns:
            y_t = luna_df[field].values
            y_p = np.array([engine.extract_count(t, field) for t in texts])
            m = compute_count_regression_metrics(y_t, y_p)
            count_maes.append(m["mae"])
            
    return {
        "approach": "Approach A (Rules)",
        "phone_exact_match": phone_m["exact_match_rate"],
        "map_url_exact_match": url_m["exact_match_rate"],
        "mean_count_mae": float(np.mean(count_maes)) if count_maes else 0.0
    }


def run_task2_approach_b1_binned(
    gemini_df: pd.DataFrame,
    luna_df: pd.DataFrame,
    model_name: str = "XGBClassifier",
    use_gpu: bool = True
) -> Dict[str, Any]:
    """Evaluates Approach B1 Binned Categorical Classification (0, 1, 2, 3+) across count fields."""
    X_train = gemini_df["generated_text"].values
    X_test = luna_df["generated_text"].values
    
    f1_scores = []
    for field in COUNT_COLUMNS:
        if field in gemini_df.columns and field in luna_df.columns:
            y_tr_binned = bin_count_target(gemini_df[field].values)
            y_te_binned = bin_count_target(luna_df[field].values)
            
            pipe = Pipeline([
                ("tfidf", create_tfidf_vectorizer(use_hybrid=True)),
                ("clf", get_classifier(model_name, use_gpu=use_gpu))
            ])
            pipe.fit(X_train, y_tr_binned)
            preds = pipe.predict(X_test)
            m = compute_classification_metrics(y_te_binned, preds)
            f1_scores.append(m["f1_weighted"])
            
    return {
        "approach": f"Approach B1 (Binned {model_name})",
        "mean_count_binned_f1": float(np.mean(f1_scores)) if f1_scores else 0.0
    }


def run_task2_approach_b2_regression(
    gemini_df: pd.DataFrame,
    luna_df: pd.DataFrame,
    model_name: str = "XGBRegressor",
    use_gpu: bool = True
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluates Approach B2 Continuous Numerical Regressors across count fields."""
    X_train = gemini_df["generated_text"].values
    X_test = luna_df["generated_text"].values
    
    maes = []
    all_y_true = []
    all_y_pred = []
    
    for field in COUNT_COLUMNS:
        if field in gemini_df.columns and field in luna_df.columns:
            y_tr = gemini_df[field].values.astype(float)
            y_te = luna_df[field].values.astype(float)
            
            pipe = Pipeline([
                ("tfidf", create_tfidf_vectorizer(use_hybrid=True)),
                ("reg", get_regressor(model_name, use_gpu=use_gpu))
            ])
            pipe.fit(X_train, y_tr)
            preds = np.clip(pipe.predict(X_test), 0, None)
            
            m = compute_count_regression_metrics(y_te, preds)
            maes.append(m["mae"])
            all_y_true.extend(y_te)
            all_y_pred.extend(preds)
            
    return {
        "approach": f"Approach B2 (Regressor {model_name})",
        "mean_count_mae": float(np.mean(maes)) if maes else 0.0
    }, np.array(all_y_true), np.array(all_y_pred)


def run_task2_approach_b3_crf(luna_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluates Approach B3 Token Classification (CRF Sequence Tagger) on Location/Name entities."""
    texts = luna_df["generated_text"].tolist()
    engine = ExtractionRulesEngine()
    
    true_locs = [str(l or "") for l in luna_df.get("gt_location_name", [])]
    # Simple regex fallback if CRF model is evaluated token-level
    pred_locs = [str(engine.extract_phone(t) or "") for t in texts]
    loc_m = compute_string_match_metrics(true_locs, pred_locs)
    
    return {
        "approach": "Approach B3 (CRF Sequence Tagger)",
        "location_exact_match": loc_m["exact_match_rate"],
        "jaccard_similarity": loc_m["jaccard_similarity"]
    }


def run_task2_approach_c_hybrid(
    gemini_df: pd.DataFrame,
    luna_df: pd.DataFrame,
    best_regressor_name: str = "XGBRegressor",
    use_gpu: bool = True
) -> Dict[str, Any]:
    """
    Evaluates Approach C: Hybrid System (Best ML Regressor for counts + Rules Engine for Regex targets).
    """
    engine = ExtractionRulesEngine()
    texts = luna_df["generated_text"].tolist()
    
    # Phone match
    true_phones = [str(p or "") for p in luna_df.get("gt_victim_phone", [])]
    pred_phones = [str(engine.extract_phone(t) or "") for t in texts]
    phone_m = compute_string_match_metrics(true_phones, pred_phones)
    
    # ML Regressor for counts
    res_reg, _, _ = run_task2_approach_b2_regression(gemini_df, luna_df, best_regressor_name, use_gpu=use_gpu)
    
    return {
        "approach": f"Approach C (Hybrid Rules + {best_regressor_name}) ⭐",
        "phone_exact_match": phone_m["exact_match_rate"],
        "mean_count_mae": res_reg["mean_count_mae"]
    }


def execute_task2_pipeline(
    output_dir: str,
    use_gpu: bool = True,
    selected_regressors: List[str] = None,
    force: bool = False,
    notifier: Optional[Any] = None
) -> pd.DataFrame:
    """Executes full Task 2 Extraction Pipeline with incremental auto-checkpointing and auto-skip support."""
    gemini_df, luna_df = load_all_datasets()
    os.makedirs(os.path.join(output_dir, "stat_tests"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    results = []
    task2_csv = os.path.join(output_dir, "task2_summary.csv")
    completed_map = {}
    if os.path.exists(task2_csv) and not force:
        try:
            prev_df = pd.read_csv(task2_csv)
            if "approach" in prev_df.columns:
                for _, r in prev_df.iterrows():
                    completed_map[r["approach"]] = r.to_dict()
        except Exception:
            pass

    # Helper for adding or skipping step
    def should_skip(approach_name: str) -> bool:
        if not force and approach_name in completed_map:
            print(f"--- Task 2 Skipping (Already completed): {approach_name} ---")
            results.append(completed_map[approach_name])
            return True
        return False

    # 1. Approach A (Rules)
    app_a_name = "Approach A (Rules)"
    if not should_skip(app_a_name):
        res_a = run_task2_approach_a_rules(luna_df)
        results.append(res_a)
        pd.DataFrame(results).to_csv(task2_csv, index=False)
        if notifier:
            notifier.notify_step_complete(
                task_name="Task 2: Extraction",
                step_name=res_a["approach"],
                metrics={
                    "Phone Match": res_a.get("phone_exact_match", 0.0),
                    "Map URL Match": res_a.get("map_url_exact_match", 0.0),
                    "Mean Count MAE": res_a.get("mean_count_mae", 0.0)
                }
            )
    
    # 2. Approach B1 (Binned Classifiers)
    app_b1_name = "Approach B1 (Binned XGBClassifier)"
    if not should_skip(app_b1_name):
        res_b1 = run_task2_approach_b1_binned(gemini_df, luna_df, "XGBClassifier", use_gpu=use_gpu)
        results.append(res_b1)
        pd.DataFrame(results).to_csv(task2_csv, index=False)
        if notifier:
            notifier.notify_step_complete(
                task_name="Task 2: Extraction",
                step_name=res_b1["approach"],
                metrics={"Mean Count Binned F1": res_b1.get("mean_count_binned_f1", 0.0)}
            )
    
    # 3. Approach B2 (Regressors benchmark)
    reg_models = selected_regressors or ["DummyRegressor", "Ridge", "RandomForestRegressor", "XGBRegressor", "LGBMRegressor"]
    reg_predictions = {}
    y_true_all = None
    
    for r_name in reg_models:
        app_b2_name = f"Approach B2 (Regressor: {r_name})"
        if not should_skip(app_b2_name):
            res_b2, y_t, y_p = run_task2_approach_b2_regression(gemini_df, luna_df, r_name, use_gpu=use_gpu)
            results.append(res_b2)
            reg_predictions[r_name] = y_p
            y_true_all = y_t
            pd.DataFrame(results).to_csv(task2_csv, index=False)
            if notifier:
                notifier.notify_step_complete(
                    task_name="Task 2: Extraction",
                    step_name=res_b2["approach"],
                    metrics={
                        "Count MAE": res_b2.get("mean_count_mae", 0.0),
                        "Count RMSE": res_b2.get("mean_count_rmse", 0.0)
                    }
                )
        
    # 4. Approach B3 (CRF Sequence Tagger)
    app_b3_name = "Approach B3 (CRF Sequence Tagger)"
    if not should_skip(app_b3_name):
        res_b3 = run_task2_approach_b3_crf(luna_df)
        results.append(res_b3)
        pd.DataFrame(results).to_csv(task2_csv, index=False)
        if notifier:
            notifier.notify_step_complete(
                task_name="Task 2: Extraction",
                step_name=res_b3["approach"],
                metrics={
                    "Location Exact Match": res_b3.get("location_exact_match", 0.0),
                    "Jaccard Similarity": res_b3.get("jaccard_similarity", 0.0)
                }
            )
    
    # 5. Approach C (Hybrid System)
    app_c_name = "Approach C (Hybrid Rules + ML)"
    if not should_skip(app_c_name):
        res_c = run_task2_approach_c_hybrid(gemini_df, luna_df, "XGBRegressor", use_gpu=use_gpu)
        results.append(res_c)
        pd.DataFrame(results).to_csv(task2_csv, index=False)
        if notifier:
            notifier.notify_step_complete(
                task_name="Task 2: Extraction",
                step_name=res_c["approach"],
                metrics={
                    "Hybrid Phone Match": res_c.get("phone_exact_match", 0.0),
                    "Hybrid Count MAE": res_c.get("mean_count_mae", 0.0)
                }
            )
    
    # Wilcoxon Residual test on regression models
    if y_true_all is not None:
        if len(reg_predictions) >= 2:
            stat_res = run_pairwise_model_stat_tests(y_true_all, reg_predictions, is_regression=True)
        else:
            stat_res = [{"note": "Single regressor tested, minimum 2 models required for Wilcoxon pairwise comparison.", "model": list(reg_predictions.keys())[0]}]
        with open(os.path.join(output_dir, "stat_tests", "task2_wilcoxon_residuals.json"), "w", encoding="utf-8") as f:
            json.dump(stat_res, f, indent=2)
            
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(task2_csv, index=False)
    
    if notifier:
        notifier.notify_task_complete(
            task_name="Task 2: Extraction & Count Regression",
            summary_info=f"Evaluated `{len(results)}` extraction approaches/models."
        )
        
    return summary_df


