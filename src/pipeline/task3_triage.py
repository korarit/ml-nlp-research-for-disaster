"""
Task 3: People Extraction & Clinical Triage Classification Pipeline.
Evaluates Task 3.1 (People Extraction), Task 3.2 (Pediatric Triage: Child <= 12),
and Task 3.3 (Adult Triage: Age > 12) using Clinical Metrics, QWK, Under-Triage Rate with 95% CI,
and McNemar + Holm-Bonferroni Stat Tests.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.utils.data_loader import load_all_datasets, extract_triage_data_by_age, DEFAULT_TRAIN_PATH, DEFAULT_TEST_PATH
from src.features.text_vectorizer import create_tfidf_vectorizer
from src.models.classifiers import get_classifier, ALL_CLASSIFIER_NAMES
from src.models.rules_engine import PediatricIITTRules, AdultIITTRules, ClauseSplitterRules
from src.utils.metrics import compute_triage_clinical_metrics, compute_string_match_metrics
from src.utils.statistical_tests import run_pairwise_model_stat_tests
from src.utils.visualization import plot_02_model_performance_comparison


def run_subtask_3_1_people_extraction(test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Subtask 3.1: Rules-based People Entity & Symptom Literal Extraction benchmark.
    """
    splitter = ClauseSplitterRules()
    exact_symptom_matches = 0
    total_victims = 0
    
    for _, row in test_df.iterrows():
        text = row.get("generated_text", "")
        extracted_clauses = splitter.extract_clauses(text)
        victims = extract_triage_data_by_age(pd.DataFrame([row]))[0]  # pediatric check
        victims_adult = extract_triage_data_by_age(pd.DataFrame([row]))[1]
        
        all_vic = pd.concat([victims, victims_adult], ignore_index=True)
        for _, vic in all_vic.iterrows():
            total_victims += 1
            gt_sym = vic.get("symptoms_literal", "").strip().lower()
            if any(gt_sym in cl.lower() for cl in extracted_clauses):
                exact_symptom_matches += 1
                
    acc = (exact_symptom_matches / total_victims) if total_victims > 0 else 1.0
    return {
        "model": "ClauseSplitterRules",
        "task_sub": "3.1_people_extraction",
        "symptom_extraction_accuracy": acc,
        "total_victims_evaluated": total_victims
    }


def run_triage_cv_and_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    is_pediatric: bool = True,
    use_gpu: bool = True
) -> Tuple[Dict[str, Any], np.ndarray]:
    if len(y_tr) < 10 or len(y_te) < 10:
        return {"model": model_name, "f1_weighted": 0.0, "critical_under_triage_rate": 0.0}, np.array([])
        
    if model_name in ["PediatricIITTRules", "AdultIITTRules"]:
        engine = PediatricIITTRules() if is_pediatric else AdultIITTRules()
        test_preds = engine.predict(list(X_te))
    else:
        vectorizer = create_tfidf_vectorizer(use_hybrid=True)
        clf = get_classifier(model_name, use_gpu=use_gpu)
        pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        test_preds = pipe.predict(X_te)
        
    metrics = compute_triage_clinical_metrics(list(y_te), list(test_preds))
    metrics["model"] = model_name
    metrics["task"] = "Pediatric Triage (3.2)" if is_pediatric else "Adult Triage (3.3)"
    
    return metrics, np.array(test_preds)


# Legacy alias for backwards compatibility
run_triage_cv_and_luna = run_triage_cv_and_test


def execute_task3_pipeline(
    output_dir: str,
    use_gpu: bool = True,
    selected_classifiers: List[str] = None,
    force: bool = False,
    notifier: Optional[Any] = None,
    train_path: str = DEFAULT_TRAIN_PATH,
    test_path: str = DEFAULT_TEST_PATH
) -> pd.DataFrame:
    """Executes full Task 3 Clinical Triage Pipeline with incremental auto-checkpointing and auto-skip support."""
    train_df, test_df = load_all_datasets(train_path=train_path, test_path=test_path)
    
    train_pedia, train_adult = extract_triage_data_by_age(train_df)
    test_pedia, test_adult = extract_triage_data_by_age(test_df)
    
    models_to_test = selected_classifiers or ["DummyClassifier", "LogisticRegression", "RandomForestClassifier", "XGBClassifier", "LGBMClassifier", "CatBoostClassifier"]
    
    results = []
    task3_csv = os.path.join(output_dir, "task3_summary.csv")
    completed_map = {}
    if os.path.exists(task3_csv) and not force:
        try:
            prev_df = pd.read_csv(task3_csv)
            if "model" in prev_df.columns:
                for _, r in prev_df.iterrows():
                    completed_map[r["model"]] = r.to_dict()
        except Exception:
            pass

    def should_skip(model_key: str) -> bool:
        if not force and model_key in completed_map:
            print(f"--- Task 3 Skipping (Already completed): {model_key} ---")
            results.append(completed_map[model_key])
            return True
        return False
    
    os.makedirs(os.path.join(output_dir, "stat_tests"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # -------------------------------------------------------------
    # Sub-task 3.1: People Extraction (Separating Victims & Symptoms)
    # -------------------------------------------------------------
    key_3_1 = "Subtask_3.1_Rule_People_Extraction"
    if not should_skip(key_3_1):
        res_3_1 = run_subtask_3_1_people_extraction(test_df)
        results.append(res_3_1)
        pd.DataFrame(results).to_csv(task3_csv, index=False)
        if notifier:
            notifier.notify_step_complete(
                task_name="Task 3: Triage",
                step_name="Sub-task 3.1: People Extraction",
                metrics={
                    "Mean Victims/Tweet": res_3_1.get("mean_extracted_victims_per_tweet", 0.0),
                    "F1 Weighted": res_3_1.get("f1_weighted", 0.0)
                }
            )
    
    # -------------------------------------------------------------
    # Sub-task 3.2: Pediatric Triage Classification (Child <= 12)
    # -------------------------------------------------------------
    pedia_models = ["PediatricIITTRules"] + models_to_test
    pedia_preds_dict = {}
    y_te_pedia = test_pedia["triage_color"].values if len(test_pedia) > 0 else np.array([])
    
    for m_name in pedia_models:
        key_3_2 = f"Pediatric_{m_name}"
        if not should_skip(key_3_2):
            m_res, preds = run_triage_cv_and_test(train_pedia, test_pedia, m_name, is_pediatric=True, use_gpu=use_gpu)
            results.append(m_res)
            if len(preds) > 0:
                pedia_preds_dict[m_name] = preds
            pd.DataFrame(results).to_csv(task3_csv, index=False)
            if notifier:
                notifier.notify_step_complete(
                    task_name="Task 3.2: Pediatric Triage",
                    step_name=f"Pediatric ({m_name})",
                    metrics={
                        "F1 Weighted": m_res.get("f1_weighted", 0.0),
                        "Triage Accuracy": m_res.get("triage_accuracy", 0.0),
                        "Under-triage Rate": m_res.get("under_triage_rate", 0.0),
                        "Critical Under-triage Rate": m_res.get("critical_under_triage_rate", 0.0)
                    }
                )
            
    if len(y_te_pedia) > 0:
        if len(pedia_preds_dict) >= 2:
            stat_pedia = run_pairwise_model_stat_tests(y_te_pedia, pedia_preds_dict, is_regression=False)
        else:
            stat_pedia = [{"note": "Single model tested for Pediatric triage, minimum 2 models required for McNemar comparison."}]
        with open(os.path.join(output_dir, "stat_tests", "task3_2_pedia_mcnemar.json"), "w", encoding="utf-8") as f:
            json.dump(stat_pedia, f, indent=2)
            
    # -------------------------------------------------------------
    # Sub-task 3.3: Adult Triage Classification (Age > 12)
    # -------------------------------------------------------------
    adult_models = ["AdultIITTRules"] + models_to_test
    adult_preds_dict = {}
    y_te_adult = test_adult["triage_color"].values if len(test_adult) > 0 else np.array([])
    
    for m_name in adult_models:
        key_3_3 = f"Adult_{m_name}"
        if not should_skip(key_3_3):
            m_res, preds = run_triage_cv_and_test(train_adult, test_adult, m_name, is_pediatric=False, use_gpu=use_gpu)
            results.append(m_res)
            if len(preds) > 0:
                adult_preds_dict[m_name] = preds
            pd.DataFrame(results).to_csv(task3_csv, index=False)
            if notifier:
                notifier.notify_step_complete(
                    task_name="Task 3.3: Adult Triage",
                    step_name=f"Adult ({m_name})",
                    metrics={
                        "F1 Weighted": m_res.get("f1_weighted", 0.0),
                        "Triage Accuracy": m_res.get("triage_accuracy", 0.0),
                        "Under-triage Rate": m_res.get("under_triage_rate", 0.0),
                        "Critical Under-triage Rate": m_res.get("critical_under_triage_rate", 0.0)
                    }
                )
            
    if len(y_te_adult) > 0:
        if len(adult_preds_dict) >= 2:
            stat_adult = run_pairwise_model_stat_tests(y_te_adult, adult_preds_dict, is_regression=False)
        else:
            stat_adult = [{"note": "Single model tested for Adult triage, minimum 2 models required for McNemar comparison."}]
        with open(os.path.join(output_dir, "stat_tests", "task3_3_adult_mcnemar.json"), "w", encoding="utf-8") as f:
            json.dump(stat_adult, f, indent=2)
            
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(task3_csv, index=False)
    
    os.makedirs(os.path.join(output_dir, "graphs"), exist_ok=True)
    renamed_df = summary_df.rename(columns={"f1_weighted": "f1", "triage_accuracy": "accuracy"})
    plot_02_model_performance_comparison(
        renamed_df,
        os.path.join(output_dir, "graphs", "02_task3_triage_comparison.png"),
        task_name="Task 3 Clinical Triage"
    )
    
    if notifier:
        notifier.notify_task_complete(
            task_name="Task 3: Clinical Triage Classification",
            summary_info=f"Evaluated Pediatric and Adult Triage models across `{len(results)}` configurations."
        )
        
    return summary_df
