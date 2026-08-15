"""
Task 1: Disaster Tweet Classification Pipeline.
Evaluates 17 Classical ML Models & Baselines using 5-Fold Stratified CV, Optuna Auto-Tuning,
Luna Held-out Test Benchmark, McNemar + Holm Stat Tests, and GPU/CPU Latency Benchmarks.
"""

import os
import json
import numpy as np
import pandas as pd
import optuna
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.utils.data_loader import load_all_datasets, DEFAULT_TRAIN_PATH, DEFAULT_TEST_PATH
from src.features.text_vectorizer import create_tfidf_vectorizer
from src.models.classifiers import get_classifier, ALL_CLASSIFIER_NAMES
from src.models.rules_engine import SimpleKeywordRules
from src.utils.metrics import compute_classification_metrics
from src.utils.latency import measure_inference_latency
from src.utils.statistical_tests import run_pairwise_model_stat_tests
from src.utils.visualization import (
    plot_01_optimization_history, plot_02_model_performance_comparison,
    plot_03_f1_f2_vs_latency_tradeoff, plot_04_confusion_matrix_heatmap, plot_06_gemini_val_vs_luna_test
)

# Silence optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_task1_cv(
    train_df: pd.DataFrame,
    model_name: str,
    hyperparams: Optional[Dict[str, Any]] = None,
    n_splits: int = 5,
    use_gpu: bool = True
) -> Dict[str, Any]:
    """Runs 5-Fold Stratified Cross-Validation for Task 1 with strict leakage prevention."""
    X = train_df["generated_text"].values
    y = train_df["gt_is_help_request_num"].astype(int).values
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    
    if model_name == "SimpleKeywordRules":
        rule_engine = SimpleKeywordRules()
        preds = rule_engine.predict(list(X))
        m = compute_classification_metrics(y, np.array(preds))
        return {
            "model": model_name,
            "mean_f1": m["f1"], "std_f1": 0.0,
            "mean_f2": m["f2"], "std_f2": 0.0,
            "mean_accuracy": m["accuracy"], "mean_precision": m["precision"],
            "mean_recall": m["recall"], "mean_mcc": m["mcc"]
        }
        
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        vectorizer = create_tfidf_vectorizer(use_hybrid=True)
        clf_params = hyperparams or {}
        clf = get_classifier(model_name, use_gpu=use_gpu, **clf_params)
        
        pipe = Pipeline([
            ("tfidf", vectorizer),
            ("clf", clf)
        ])
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        
        m = compute_classification_metrics(y_val, y_pred)
        fold_metrics.append(m)
        
    df_m = pd.DataFrame(fold_metrics)
    return {
        "model": model_name,
        "mean_f1": float(df_m["f1"].mean()), "std_f1": float(df_m["f1"].std()),
        "mean_f2": float(df_m["f2"].mean()), "std_f2": float(df_m["f2"].std()),
        "mean_accuracy": float(df_m["accuracy"].mean()),
        "mean_precision": float(df_m["precision"].mean()),
        "mean_recall": float(df_m["recall"].mean()),
        "mean_mcc": float(df_m["mcc"].mean())
    }


def optimize_task1_model(
    train_df: pd.DataFrame,
    model_name: str,
    n_trials: int = 15,
    use_gpu: bool = True
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Runs Optuna hyperparameter optimization over 5-Fold CV for a specific model."""
    trials_log = []
    
    def objective(trial):
        params = {}
        if model_name in ["XGBClassifier", "LGBMClassifier", "CatBoostClassifier"]:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 250)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
        elif model_name in ["LogisticRegression", "LinearSVC", "PassiveAggressiveClassifier"]:
            params["C"] = trial.suggest_float("C", 0.01, 10.0, log=True)
        elif model_name in ["SVC_linear", "SVC_rbf", "SVC_poly", "SVC_sigmoid", "SVC"]:
            params["C"] = trial.suggest_float("C", 0.01, 10.0, log=True)
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
        elif model_name == "RidgeClassifier":
            params["alpha"] = trial.suggest_float("alpha", 0.01, 10.0, log=True)
        elif model_name == "SGDClassifier":
            params["alpha"] = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
            params["loss"] = trial.suggest_categorical("loss", ["hinge", "log_loss", "modified_huber"])
        elif model_name in ["MultinomialNB", "ComplementNB"]:
            params["alpha"] = trial.suggest_float("alpha", 0.01, 5.0)
        elif model_name == "KNeighborsClassifier":
            params["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 15)
            params["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
        elif model_name == "MLPClassifier":
            params["alpha"] = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
            params["learning_rate_init"] = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
        elif model_name in ["RandomForestClassifier", "ExtraTreesClassifier"]:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 200)
            params["max_depth"] = trial.suggest_int("max_depth", 5, 25)
        elif model_name == "DecisionTreeClassifier":
            params["max_depth"] = trial.suggest_int("max_depth", 3, 20)
        elif model_name == "AdaBoostClassifier":
            params["n_estimators"] = trial.suggest_int("n_estimators", 30, 150)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 1.0, log=True)
        elif model_name == "GradientBoostingClassifier":
            params["n_estimators"] = trial.suggest_int("n_estimators", 30, 100)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 6)

        res = run_task1_cv(train_df, model_name, hyperparams=params, use_gpu=use_gpu)
        score = 0.5 * res["mean_f1"] + 0.5 * res["mean_f2"]
        trials_log.append({"trial": trial.number, "score": score, "f1": res["mean_f1"], "f2": res["mean_f2"], "params": params})
        return score
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    trials_df = pd.DataFrame(trials_log)
    return best_params, trials_df



import time


def execute_task1_pipeline(
    output_dir: str,
    selected_models: List[str] = None,
    use_gpu: bool = True,
    n_trials: int = 15,
    n_splits: int = 5,
    latency_runs: int = 1000,
    force: bool = False,
    notifier: Optional[Any] = None,
    train_path: str = DEFAULT_TRAIN_PATH,
    test_path: str = DEFAULT_TEST_PATH
) -> pd.DataFrame:
    """Executes full Task 1 Pipeline with incremental auto-checkpointing and auto-skip support."""
    train_df, test_df = load_all_datasets(train_path=train_path, test_path=test_path)
    models_to_run = selected_models or (["SimpleKeywordRules"] + ALL_CLASSIFIER_NAMES)
    
    summary_results = []
    test_predictions = {}
    best_configs = {}
    
    X_train_full = train_df["generated_text"].values
    y_train_full = train_df["gt_is_help_request_num"].astype(int).values
    X_test = test_df["generated_text"].values
    y_test = test_df["gt_is_help_request_num"].astype(int).values
    
    os.makedirs(os.path.join(output_dir, "best_configs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "graphs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "stat_tests"), exist_ok=True)
    
    summary_csv_path = os.path.join(output_dir, "task1_summary.csv")
    completed_models_data = {}
    if os.path.exists(summary_csv_path) and not force:
        try:
            prev_df = pd.read_csv(summary_csv_path)
            if "model" in prev_df.columns:
                for _, r in prev_df.iterrows():
                    m_n = r["model"]
                    completed_models_data[m_n] = r.to_dict()
        except Exception:
            pass

    for m_name in models_to_run:
        if not force and m_name in completed_models_data:
            print(f"--- Task 1 Skipping (Already completed): {m_name} ---")
            row = completed_models_data[m_name]
            summary_results.append(row)
            
            cfg_path = os.path.join(output_dir, "best_configs", f"best_params_task1_{m_name}.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    best_configs[m_name] = json.load(f)
            else:
                best_configs[m_name] = {}
                
            try:
                if m_name == "SimpleKeywordRules":
                    engine = SimpleKeywordRules()
                    test_preds = engine.predict(list(X_test))
                else:
                    vectorizer = create_tfidf_vectorizer(use_hybrid=True)
                    clf = get_classifier(m_name, use_gpu=use_gpu, **best_configs[m_name])
                    pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
                    pipe.fit(X_train_full, y_train_full)
                    test_preds = pipe.predict(X_test)
                test_predictions[m_name] = np.array(test_preds)
            except Exception as e:
                print(f"Warning: Failed to regenerate predictions for skipped model {m_name}: {e}")
            continue

        print(f"--- Task 1 Running: {m_name} ---")
        start_t = time.time()
        
        # Step 1: 5-Fold CV & Auto-tuning
        if m_name == "SimpleKeywordRules" or m_name == "DummyClassifier":
            best_p = {}
            cv_res = run_task1_cv(train_df, m_name, n_splits=n_splits, use_gpu=use_gpu)
        else:
            best_p, trials_df = optimize_task1_model(train_df, m_name, n_trials=n_trials, use_gpu=use_gpu)
            cv_res = run_task1_cv(train_df, m_name, hyperparams=best_p, n_splits=n_splits, use_gpu=use_gpu)
            
            # Save Optuna trial log for model
            trials_df.to_json(os.path.join(output_dir, "logs", f"trials_task1_{m_name}.json"), orient="records", indent=2)
            plot_01_optimization_history(trials_df, os.path.join(output_dir, "graphs", f"task1_{m_name}_opt_history.png"))
            
        best_configs[m_name] = best_p
        
        # Immediate Checkpoint: Save best config per model
        with open(os.path.join(output_dir, "best_configs", f"best_params_task1_{m_name}.json"), "w", encoding="utf-8") as f:
            json.dump(best_p, f, indent=2)
            
        # Step 2: Re-fit on full training set & Test on held-out test set
        if m_name == "SimpleKeywordRules":
            engine = SimpleKeywordRules()
            test_preds = engine.predict(list(X_test))
            predict_fn = engine.predict_one
        else:
            vectorizer = create_tfidf_vectorizer(use_hybrid=True)
            clf = get_classifier(m_name, use_gpu=use_gpu, **best_p)
            pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
            pipe.fit(X_train_full, y_train_full)
            test_preds = pipe.predict(X_test)
            predict_fn = lambda txt: pipe.predict([txt])[0]
            
        test_predictions[m_name] = np.array(test_preds)
        test_metrics = compute_classification_metrics(y_test, np.array(test_preds))
        
        # Step 3: Latency Measurement
        lat_gpu = measure_inference_latency(predict_fn, n_runs=min(latency_runs, 200), use_gpu=True) if use_gpu else {"p95_latency_ms": 0.0, "qps": 0.0}
        lat_cpu = measure_inference_latency(predict_fn, n_runs=min(latency_runs, 200), use_gpu=False)
        
        elapsed_sec = time.time() - start_t
        
        row = {
            "model": m_name,
            "cv_f1_mean": cv_res["mean_f1"], "cv_f1_std": cv_res["std_f1"],
            "cv_f2_mean": cv_res["mean_f2"], "cv_f2_std": cv_res["std_f2"],
            "luna_f1": test_metrics["f1"], "luna_f2": test_metrics["f2"],
            "luna_accuracy": test_metrics["accuracy"], "luna_precision": test_metrics["precision"],
            "luna_recall": test_metrics["recall"], "luna_mcc": test_metrics["mcc"],
            "gpu_p95_latency_ms": lat_gpu["p95_latency_ms"], "gpu_qps": lat_gpu["qps"],
            "cpu_p95_latency_ms": lat_cpu["p95_latency_ms"], "cpu_qps": lat_cpu["qps"]
        }
        summary_results.append(row)
        
        # Immediate Checkpoint: Write summary CSV after each model
        current_df = pd.DataFrame(summary_results)
        current_df.to_csv(os.path.join(output_dir, "task1_summary.csv"), index=False)
        
        # Discord Notification: 1 model train complete
        if notifier:
            notifier.notify_step_complete(
                task_name="Task 1: Classification",
                step_name=m_name,
                metrics={
                    "CV Mean F1": row["cv_f1_mean"],
                    "CV Mean F2": row["cv_f2_mean"],
                    "Test F1": row["luna_f1"],
                    "Test F2": row["luna_f2"],
                    "CPU Latency P95 (ms)": row["cpu_p95_latency_ms"]
                },
                elapsed_sec=elapsed_sec
            )
        
    summary_df = pd.DataFrame(summary_results)
    
    with open(os.path.join(output_dir, "best_configs", "task1_best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_configs, f, indent=2)
        
    # Step 4: Statistical Significance Tests (McNemar + Holm)
    if len(test_predictions) >= 2:
        stat_results = run_pairwise_model_stat_tests(y_test, test_predictions, is_regression=False)
    else:
        stat_results = [{"note": "Single model tested, minimum 2 models required for McNemar pairwise comparison.", "model": list(test_predictions.keys())[0]}]
        
    with open(os.path.join(output_dir, "stat_tests", "task1_mcnemar_holm.json"), "w", encoding="utf-8") as f:
        json.dump(stat_results, f, indent=2)
        
    # Step 5: Visualizations
    renamed_df = summary_df.rename(columns={
        "luna_f1": "f1", "luna_f2": "f2", "luna_accuracy": "accuracy",
        "luna_precision": "precision", "luna_recall": "recall", "luna_mcc": "mcc"
    })
    
    plot_02_model_performance_comparison(
        renamed_df,
        os.path.join(output_dir, "graphs", "02_task1_model_comparison.png"),
        task_name="Task 1 Classification"
    )
    
    plot_03_f1_f2_vs_latency_tradeoff(
        summary_df.rename(columns={"cpu_p95_latency_ms": "p95_latency_ms", "luna_f1": "f1", "luna_f2": "f2"}),
        os.path.join(output_dir, "graphs", "03_task1_latency_tradeoff.png")
    )
    
    gap_df = pd.DataFrame({
        "model": summary_df["model"],
        "cv_score": summary_df["cv_f1_mean"],
        "luna_score": summary_df["luna_f1"]
    })
    plot_06_gemini_val_vs_luna_test(gap_df, os.path.join(output_dir, "graphs", "06_task1_performance_gap.png"))
    
    if notifier and len(summary_df) > 0:
        best_row = summary_df.loc[summary_df["luna_f1"].idxmax()]
        notifier.notify_task_complete(
            task_name="Task 1: Classification",
            summary_info=f"Evaluated `{len(summary_df)}` models.\n🏆 **Top Classifier**: `{best_row['model']}` (Test F1: `{best_row['luna_f1']:.4f}`, F2: `{best_row['luna_f2']:.4f}`)"
        )
    
    return summary_df
