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

from src.utils.data_loader import load_all_datasets, bin_count_target, COUNT_COLUMNS, DEFAULT_TRAIN_PATH, DEFAULT_TEST_PATH
from src.features.text_vectorizer import create_tfidf_vectorizer
from src.models.classifiers import get_classifier, ALL_CLASSIFIER_NAMES
from src.models.regressors import get_regressor, ALL_REGRESSOR_NAMES
from src.models.rules_engine import ExtractionRulesEngine
from src.utils.metrics import (
    compute_count_regression_metrics, compute_string_match_metrics,
    compute_classification_metrics
)
from src.utils.statistical_tests import run_pairwise_model_stat_tests
from src.utils.visualization import plot_02_model_performance_comparison, plot_task2_extraction_comparison


def run_task2_approach_a_rules(test_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluates Approach A Rule-Based Extraction Engine on Test Dataset."""
    engine = ExtractionRulesEngine()
    texts = test_df["generated_text"].tolist()
    
    # Phone match
    true_phones = [str(p or "") for p in test_df.get("gt_victim_phone", [])]
    pred_phones = [str(engine.extract_phone(t) or "") for t in texts]
    phone_m = compute_string_match_metrics(true_phones, pred_phones)
    
    # Map URL match
    true_urls = [str(u or "") for u in test_df.get("gt_google_map_url", [])]
    pred_urls = [str(engine.extract_map_url(t) or "") for t in texts]
    url_m = compute_string_match_metrics(true_urls, pred_urls)
    
    # Location match
    true_locs = [str(l or "") for l in test_df.get("gt_location_name", [])]
    pred_locs = [str(engine.extract_location(t) or "") for t in texts]
    loc_m = compute_string_match_metrics(true_locs, pred_locs)
    
    # Coordinates match
    true_coord_strs = [f"{row.get('gt_lat', '')},{row.get('gt_lng', '')}" if pd.notna(row.get('gt_lat')) else "" for _, row in test_df.iterrows()]
    pred_coords = [engine.extract_coords(t) for t in texts]
    pred_coord_strs = [f"{c[0]},{c[1]}" if c[0] is not None else "" for c in pred_coords]
    coords_m = compute_string_match_metrics(true_coord_strs, pred_coord_strs)
    
    # Count metrics
    count_maes = []
    count_rmses = []
    count_ems = []
    count_f1s = []
    count_f2s = []
    for field in COUNT_COLUMNS:
        if field in test_df.columns:
            y_t = test_df[field].values.astype(float)
            y_p = np.array([engine.extract_count(t, field) for t in texts], dtype=float)
            m = compute_count_regression_metrics(y_t, y_p)
            count_maes.append(m["mae"])
            count_rmses.append(m["rmse"])
            count_ems.append(m["exact_match"])
            
            clf_m = compute_classification_metrics(bin_count_target(y_t), bin_count_target(y_p))
            count_f1s.append(clf_m["f1_weighted"])
            count_f2s.append(clf_m["f2"])
            
    return {
        "approach": "Approach A (Rules)",
        "f1": float(np.mean(count_f1s)) if count_f1s else 0.0,
        "f2": float(np.mean(count_f2s)) if count_f2s else 0.0,
        "phone_exact_match": phone_m["exact_match_rate"],
        "map_url_exact_match": url_m["exact_match_rate"],
        "coords_exact_match": coords_m["exact_match_rate"],
        "location_exact_match": loc_m["exact_match_rate"],
        "mean_count_mae": float(np.mean(count_maes)) if count_maes else 0.0,
        "mean_count_rmse": float(np.mean(count_rmses)) if count_rmses else 0.0,
        "count_exact_match": float(np.mean(count_ems)) if count_ems else 0.0
    }


def run_task2_approach_b1_binned(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "XGBClassifier",
    use_gpu: bool = True
) -> Dict[str, Any]:
    """Evaluates Approach B1 Binned Categorical Classification (0, 1, 2, 3+) across count fields."""
    engine = ExtractionRulesEngine()
    texts = test_df["generated_text"].tolist()
    
    # Phone, Map URL, and Location match (via Rules engine)
    true_phones = [str(p or "") for p in test_df.get("gt_victim_phone", [])]
    pred_phones = [str(engine.extract_phone(t) or "") for t in texts]
    phone_m = compute_string_match_metrics(true_phones, pred_phones)
    
    true_urls = [str(u or "") for u in test_df.get("gt_google_map_url", [])]
    pred_urls = [str(engine.extract_map_url(t) or "") for t in texts]
    url_m = compute_string_match_metrics(true_urls, pred_urls)
    
    true_locs = [str(l or "") for l in test_df.get("gt_location_name", [])]
    pred_locs = [str(engine.extract_location(t) or "") for t in texts]
    loc_m = compute_string_match_metrics(true_locs, pred_locs)

    true_coord_strs = [f"{row.get('gt_lat', '')},{row.get('gt_lng', '')}" if pd.notna(row.get('gt_lat')) else "" for _, row in test_df.iterrows()]
    pred_coords = [engine.extract_coords(t) for t in texts]
    pred_coord_strs = [f"{c[0]},{c[1]}" if c[0] is not None else "" for c in pred_coords]
    coords_m = compute_string_match_metrics(true_coord_strs, pred_coord_strs)

    X_train = train_df["generated_text"].values
    X_test = test_df["generated_text"].values
    
    maes = []
    rmses = []
    ems = []
    f1s = []
    f2s = []
    for field in COUNT_COLUMNS:
        if field in train_df.columns and field in test_df.columns:
            y_tr_raw = train_df[field].values.astype(float)
            y_te_raw = test_df[field].values.astype(float)
            y_tr_binned = bin_count_target(y_tr_raw)
            y_te_binned = bin_count_target(y_te_raw)
            
            pipe = Pipeline([
                ("tfidf", create_tfidf_vectorizer(use_hybrid=True)),
                ("clf", get_classifier(model_name, use_gpu=use_gpu))
            ])
            pipe.fit(X_train, y_tr_binned)
            preds_binned = pipe.predict(X_test)
            
            m = compute_count_regression_metrics(y_te_binned, preds_binned)
            clf_m = compute_classification_metrics(y_te_binned, preds_binned)
            maes.append(m["mae"])
            rmses.append(m["rmse"])
            ems.append(m["exact_match"])
            f1s.append(clf_m["f1_weighted"])
            f2s.append(clf_m["f2"])
            
    return {
        "approach": f"Approach B1 (Binned {model_name})",
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "f2": float(np.mean(f2s)) if f2s else 0.0,
        "phone_exact_match": phone_m["exact_match_rate"],
        "map_url_exact_match": url_m["exact_match_rate"],
        "coords_exact_match": coords_m["exact_match_rate"],
        "location_exact_match": loc_m["exact_match_rate"],
        "mean_count_mae": float(np.mean(maes)) if maes else 0.0,
        "mean_count_rmse": float(np.mean(rmses)) if rmses else 0.0,
        "count_exact_match": float(np.mean(ems)) if ems else 0.0
    }


def run_task2_approach_b2_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "XGBRegressor",
    use_gpu: bool = True
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluates Approach B2 Continuous Numerical Regressors across count fields."""
    engine = ExtractionRulesEngine()
    texts = test_df["generated_text"].tolist()
    
    # Phone, Map URL, and Location match (via Rules engine)
    true_phones = [str(p or "") for p in test_df.get("gt_victim_phone", [])]
    pred_phones = [str(engine.extract_phone(t) or "") for t in texts]
    phone_m = compute_string_match_metrics(true_phones, pred_phones)
    
    true_urls = [str(u or "") for u in test_df.get("gt_google_map_url", [])]
    pred_urls = [str(engine.extract_map_url(t) or "") for t in texts]
    url_m = compute_string_match_metrics(true_urls, pred_urls)
    
    true_locs = [str(l or "") for l in test_df.get("gt_location_name", [])]
    pred_locs = [str(engine.extract_location(t) or "") for t in texts]
    loc_m = compute_string_match_metrics(true_locs, pred_locs)

    true_coord_strs = [f"{row.get('gt_lat', '')},{row.get('gt_lng', '')}" if pd.notna(row.get('gt_lat')) else "" for _, row in test_df.iterrows()]
    pred_coords = [engine.extract_coords(t) for t in texts]
    pred_coord_strs = [f"{c[0]},{c[1]}" if c[0] is not None else "" for c in pred_coords]
    coords_m = compute_string_match_metrics(true_coord_strs, pred_coord_strs)

    X_train = train_df["generated_text"].values
    X_test = test_df["generated_text"].values
    
    maes = []
    rmses = []
    ems = []
    f1s = []
    f2s = []
    all_y_true = []
    all_y_pred = []
    
    for field in COUNT_COLUMNS:
        if field in train_df.columns and field in test_df.columns:
            y_tr = train_df[field].values.astype(float)
            y_te = test_df[field].values.astype(float)
            
            pipe = Pipeline([
                ("tfidf", create_tfidf_vectorizer(use_hybrid=True)),
                ("reg", get_regressor(model_name, use_gpu=use_gpu))
            ])
            pipe.fit(X_train, y_tr)
            preds = np.clip(pipe.predict(X_test), 0, None)
            
            m = compute_count_regression_metrics(y_te, preds)
            clf_m = compute_classification_metrics(bin_count_target(y_te), bin_count_target(preds))
            maes.append(m["mae"])
            rmses.append(m["rmse"])
            ems.append(m["exact_match"])
            f1s.append(clf_m["f1_weighted"])
            f2s.append(clf_m["f2"])
            all_y_true.extend(y_te)
            all_y_pred.extend(preds)
            
    return {
        "approach": f"Approach B2 (Regressor {model_name})",
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "f2": float(np.mean(f2s)) if f2s else 0.0,
        "phone_exact_match": phone_m["exact_match_rate"],
        "map_url_exact_match": url_m["exact_match_rate"],
        "coords_exact_match": coords_m["exact_match_rate"],
        "location_exact_match": loc_m["exact_match_rate"],
        "mean_count_mae": float(np.mean(maes)) if maes else 0.0,
        "mean_count_rmse": float(np.mean(rmses)) if rmses else 0.0,
        "count_exact_match": float(np.mean(ems)) if ems else 0.0
    }, np.array(all_y_true), np.array(all_y_pred)


def run_task2_approach_b3_crf(test_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluates Approach B3 Token Classification (CRF Sequence Tagger) on Location/Name entities."""
    texts = test_df["generated_text"].tolist()
    engine = ExtractionRulesEngine()
    
    true_phones = [str(p or "") for p in test_df.get("gt_victim_phone", [])]
    pred_phones = [str(engine.extract_phone(t) or "") for t in texts]
    phone_m = compute_string_match_metrics(true_phones, pred_phones)

    true_urls = [str(u or "") for u in test_df.get("gt_google_map_url", [])]
    pred_urls = [str(engine.extract_map_url(t) or "") for t in texts]
    url_m = compute_string_match_metrics(true_urls, pred_urls)
    
    true_locs = [str(l or "") for l in test_df.get("gt_location_name", [])]
    pred_locs = [str(engine.extract_location(t) or "") for t in texts]
    loc_m = compute_string_match_metrics(true_locs, pred_locs)

    true_coord_strs = [f"{row.get('gt_lat', '')},{row.get('gt_lng', '')}" if pd.notna(row.get('gt_lat')) else "" for _, row in test_df.iterrows()]
    pred_coords = [engine.extract_coords(t) for t in texts]
    pred_coord_strs = [f"{c[0]},{c[1]}" if c[0] is not None else "" for c in pred_coords]
    coords_m = compute_string_match_metrics(true_coord_strs, pred_coord_strs)
    
    return {
        "approach": "Approach B3 (CRF Sequence Tagger)",
        "f1": loc_m["exact_match_rate"],
        "f2": loc_m["exact_match_rate"],
        "phone_exact_match": phone_m["exact_match_rate"],
        "map_url_exact_match": url_m["exact_match_rate"],
        "coords_exact_match": coords_m["exact_match_rate"],
        "location_exact_match": loc_m["exact_match_rate"],
        "mean_count_mae": 0.0,
        "mean_count_rmse": 0.0,
        "count_exact_match": 0.0
    }


def run_task2_approach_c_hybrid(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_regressor_name: str = "XGBRegressor",
    use_gpu: bool = True
) -> Dict[str, Any]:
    """
    Evaluates Approach C: Hybrid System (Best ML Regressor for counts + Rules Engine for Regex targets).
    """
    res_a = run_task2_approach_a_rules(test_df)
    res_reg, _, _ = run_task2_approach_b2_regression(train_df, test_df, best_regressor_name, use_gpu=use_gpu)
    
    return {
        "approach": f"Approach C (Hybrid Rules + {best_regressor_name}) ⭐",
        "f1": res_reg["f1"],
        "f2": res_reg["f2"],
        "phone_exact_match": res_a["phone_exact_match"],
        "map_url_exact_match": res_a["map_url_exact_match"],
        "coords_exact_match": res_a["coords_exact_match"],
        "location_exact_match": res_a["location_exact_match"],
        "mean_count_mae": res_reg["mean_count_mae"],
        "mean_count_rmse": res_reg["mean_count_rmse"],
        "count_exact_match": res_reg["count_exact_match"]
    }


def execute_task2_pipeline(
    output_dir: str,
    use_gpu: bool = True,
    selected_regressors: List[str] = None,
    force: bool = False,
    notifier: Optional[Any] = None,
    train_path: str = DEFAULT_TRAIN_PATH,
    test_path: str = DEFAULT_TEST_PATH
) -> pd.DataFrame:
    """Executes full Task 2 Extraction Pipeline with incremental auto-checkpointing and auto-skip support."""
    train_df, test_df = load_all_datasets(train_path=train_path, test_path=test_path)
    os.makedirs(os.path.join(output_dir, "stat_tests"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "graphs"), exist_ok=True)
    
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

    def notify_step(res_dict: Dict[str, Any]):
        if notifier:
            notifier.notify_step_complete(
                task_name="Task 2: Extraction",
                step_name=res_dict["approach"],
                metrics={
                    "F1-Score": res_dict.get("f1", 0.0),
                    "F2-Score": res_dict.get("f2", 0.0),
                    "Phone Match Rate": res_dict.get("phone_exact_match", 0.0),
                    "Location Match Rate": res_dict.get("location_exact_match", 0.0),
                    "Map URL Match Rate": res_dict.get("map_url_exact_match", 0.0),
                    "Coordinates Match Rate": res_dict.get("coords_exact_match", 0.0),
                    "Count Exact Match Rate": res_dict.get("count_exact_match", 0.0),
                    "Count MAE": res_dict.get("mean_count_mae", 0.0),
                    "Count RMSE": res_dict.get("mean_count_rmse", 0.0)
                }
            )

    # 1. Approach A (Rules)
    app_a_name = "Approach A (Rules)"
    if not should_skip(app_a_name):
        res_a = run_task2_approach_a_rules(test_df)
        results.append(res_a)
        pd.DataFrame(results).to_csv(task2_csv, index=False)
        notify_step(res_a)
    
    # 2. Approach B1 (Binned Classifiers)
    app_b1_name = "Approach B1 (Binned XGBClassifier)"
    if not should_skip(app_b1_name):
        res_b1 = run_task2_approach_b1_binned(train_df, test_df, "XGBClassifier", use_gpu=use_gpu)
        results.append(res_b1)
        pd.DataFrame(results).to_csv(task2_csv, index=False)
        notify_step(res_b1)
    
    # 3. Approach B2 (Regressors benchmark)
    reg_models = selected_regressors or ["DummyRegressor", "Ridge", "RandomForestRegressor", "XGBRegressor", "LGBMRegressor"]
    reg_predictions = {}
    y_true_all = None
    
    for r_name in reg_models:
        app_b2_name = f"Approach B2 (Regressor {r_name})"
        if not should_skip(app_b2_name):
            res_b2, y_t, y_p = run_task2_approach_b2_regression(train_df, test_df, r_name, use_gpu=use_gpu)
            results.append(res_b2)
            reg_predictions[r_name] = y_p
            y_true_all = y_t
            pd.DataFrame(results).to_csv(task2_csv, index=False)
            notify_step(res_b2)
        
    # 4. Approach B3 (CRF Sequence Tagger)
    app_b3_name = "Approach B3 (CRF Sequence Tagger)"
    if not should_skip(app_b3_name):
        res_b3 = run_task2_approach_b3_crf(test_df)
        results.append(res_b3)
        pd.DataFrame(results).to_csv(task2_csv, index=False)
        notify_step(res_b3)
    
    # 5. Approach C (Hybrid System)
    app_c_name = "Approach C (Hybrid Rules + XGBRegressor) ⭐"
    if not should_skip(app_c_name):
        res_c = run_task2_approach_c_hybrid(train_df, test_df, "XGBRegressor", use_gpu=use_gpu)
        results.append(res_c)
        pd.DataFrame(results).to_csv(task2_csv, index=False)
        notify_step(res_c)
    
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
    
    # Visualizations / Graph generation
    plot_02_model_performance_comparison(
        summary_df,
        os.path.join(output_dir, "graphs", "02_task2_extraction_comparison.png"),
        task_name="Task 2 Extraction"
    )
    plot_task2_extraction_comparison(
        summary_df,
        os.path.join(output_dir, "graphs")
    )
    
    if notifier:
        notifier.notify_task_complete(
            task_name="Task 2: Extraction & Count Regression",
            summary_info=f"Evaluated `{len(results)}` extraction approaches/models with standardized metrics."
        )
        
    return summary_df
