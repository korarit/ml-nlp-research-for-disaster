"""
Task 4: End-to-End Integrated Classical ML Pipeline Benchmark (Section 5.2 & Task 4).
Integrates Task 1 -> Task 2 -> Task 3.1 -> Task 3.2 & Task 3.3 into a unified streaming flow.
Evaluates overall JSON alignment, E2E clinical triage accuracy, under-triage rate (95% CI),
and cumulative E2E Latency (GPU & CPU) on Luna Held-out Test Set.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.pipeline import Pipeline

from src.utils.data_loader import load_all_datasets, extract_triage_data_by_age
from src.features.text_vectorizer import create_tfidf_vectorizer
from src.models.classifiers import get_classifier
from src.models.regressors import get_regressor
from src.models.rules_engine import (
    SimpleKeywordRules, ExtractionRulesEngine, PediatricIITTRules, AdultIITTRules, ClauseSplitterRules
)
from src.utils.metrics import (
    compute_classification_metrics, compute_triage_clinical_metrics,
    compute_string_match_metrics
)
from src.utils.latency import measure_inference_latency


class FullIntegratedDisasterPipeline:
    """
    End-to-End Integrated Disaster NLP Streaming Pipeline.
    """
    
    def __init__(self, use_gpu: bool = True, task1_model_name: str = "XGBClassifier", task3_model_name: str = "XGBClassifier"):
        self.use_gpu = use_gpu
        self.rules_engine = ExtractionRulesEngine()
        self.clause_splitter = ClauseSplitterRules()
        self.pedia_rules = PediatricIITTRules()
        self.adult_rules = AdultIITTRules()
        
        # Step 1: Task 1 Classifier
        self.task1_pipe = Pipeline([
            ("tfidf", create_tfidf_vectorizer(use_hybrid=True)),
            ("clf", get_classifier(task1_model_name, use_gpu=use_gpu))
        ])
        
        # Step 3: Triage Classifiers
        self.pedia_triage_pipe = Pipeline([
            ("tfidf", create_tfidf_vectorizer(use_hybrid=True)),
            ("clf", get_classifier(task3_model_name, use_gpu=use_gpu))
        ])
        self.adult_triage_pipe = Pipeline([
            ("tfidf", create_tfidf_vectorizer(use_hybrid=True)),
            ("clf", get_classifier(task3_model_name, use_gpu=use_gpu))
        ])
        self.is_fitted = False
        
    def fit(self, train_df: pd.DataFrame):
        """Fits Task 1 and Task 3 models on training set."""
        X_tr = train_df["generated_text"].values
        y_tr_help = train_df["gt_is_help_request_num"].values
        self.task1_pipe.fit(X_tr, y_tr_help)
        
        # Extract triage training samples
        pedia_df, adult_df = extract_triage_data_by_age(train_df)
        
        if len(pedia_df) > 5:
            self.pedia_triage_pipe.fit(pedia_df["symptoms_literal"].values, pedia_df["triage_color"].values)
        if len(adult_df) > 5:
            self.adult_triage_pipe.fit(adult_df["symptoms_literal"].values, adult_df["triage_color"].values)
            
        self.is_fitted = True
        return self
        
    def process_tweet(self, text: str) -> Dict[str, Any]:
        """
        Executes Section 5.2 E2E Flow for 1 raw tweet:
        Step 1: Classification (Task 1)
        Step 2: Rules & Count Extraction (Task 2)
        Step 3: Individual Victim & Symptom Extraction (Task 3.1)
        Step 4: Age-based Triage Routing (Task 3.2 Pediatric vs Task 3.3 Adult)
        """
        s_text = str(text)
        
        # Step 1: Disaster Classification
        is_help = int(self.task1_pipe.predict([s_text])[0]) if self.is_fitted else SimpleKeywordRules().predict_one(s_text)
        category = "help_request" if is_help == 1 else "other"
        
        if category == "other":
            return {
                "classification_category": "other",
                "is_help_request": False,
                "extracted_entities": {},
                "counts": {},
                "victims": []
            }
            
        # Step 2: Extraction Rules
        phone = self.rules_engine.extract_phone(s_text)
        lat, lng = self.rules_engine.extract_coords(s_text)
        map_url = self.rules_engine.extract_map_url(s_text)
        counts = self.rules_engine.extract_all_counts(s_text)
        
        # Step 3: Task 3.1 People Extractor
        parsed_victims = self.clause_splitter.extract_victims(s_text)
        victims_output = []
        
        # Step 4: Age-based Triage Routing per victim
        for vic in parsed_victims:
            symp = vic["symptoms_literal"]
            is_child = vic.get("is_child", False)
            
            if is_child:
                if self.is_fitted:
                    try:
                        color = str(self.pedia_triage_pipe.predict([symp])[0])
                    except Exception:
                        color = self.pedia_rules.predict_one(symp)
                else:
                    color = self.pedia_rules.predict_one(symp)
            else:
                if self.is_fitted:
                    try:
                        color = str(self.adult_triage_pipe.predict([symp])[0])
                    except Exception:
                        color = self.adult_rules.predict_one(symp)
                else:
                    color = self.adult_rules.predict_one(symp)
                    
            victims_output.append({
                "name": vic.get("name"),
                "age": vic.get("age"),
                "age_group": vic.get("age_group"),
                "triage_color": color,
                "symptoms_literal": symp
            })
            
        return {
            "classification_category": "help_request",
            "is_help_request": True,
            "extracted_entities": {
                "phone": phone,
                "lat": lat,
                "lng": lng,
                "map_url": map_url
            },
            "counts": counts,
            "victims": victims_output
        }



def execute_task4_e2e_pipeline(
    output_dir: str,
    use_gpu: bool = True,
    latency_runs: int = 1000,
    force: bool = False,
    notifier: Optional[Any] = None
) -> Dict[str, Any]:
    """Executes full Task 4 End-to-End Integrated Pipeline Benchmark on held-out test dataset with auto-skip support."""
    train_df, test_df = load_all_datasets()
    
    out_json_path = os.path.join(output_dir, "task4_e2e_results", "task4_e2e_metrics.json")
    if os.path.exists(out_json_path) and not force:
        print("--- Task 4 Skipping (Already completed): Full Integrated Pipeline Benchmark ---")
        try:
            with open(out_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    os.makedirs(os.path.join(output_dir, "task4_e2e_results"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "graphs"), exist_ok=True)
    
    pipeline = FullIntegratedDisasterPipeline(use_gpu=use_gpu)
    pipeline.fit(train_df)
    
    texts_test = test_df["generated_text"].tolist()
    y_true_help = test_df["gt_is_help_request_num"].values
    
    # Run E2E Flow on held-out test set
    e2e_outputs = [pipeline.process_tweet(t) for t in texts_test]
    y_pred_help = np.array([1 if out["is_help_request"] else 0 for out in e2e_outputs])
    
    # 1. Classification Metrics
    task1_metrics = compute_classification_metrics(y_true_help, y_pred_help)
    
    # 2. Clinical Triage Metrics
    _, test_adult = extract_triage_data_by_age(test_df)
    true_triages = test_adult["triage_color"].tolist() if len(test_adult) > 0 else ["GREEN"] * len(texts_test)
    pred_triages = [out["victims"][0]["triage_color"] if out["victims"] else "GREEN" for out in e2e_outputs[:len(true_triages)]]
    
    clinical_metrics = compute_triage_clinical_metrics(true_triages, pred_triages)
    
    # 3. Latency Benchmarks (GPU & CPU)
    lat_gpu = measure_inference_latency(pipeline.process_tweet, n_runs=min(latency_runs, 200), use_gpu=True) if use_gpu else {"p95_latency_ms": 0.0, "qps": 0.0}
    lat_cpu = measure_inference_latency(pipeline.process_tweet, n_runs=min(latency_runs, 200), use_gpu=False)
    
    e2e_results = {
        "overall_e2e_f1": task1_metrics["f1"],
        "overall_e2e_f2": task1_metrics["f2"],
        "overall_e2e_accuracy": task1_metrics["accuracy"],
        "clinical_qwk": clinical_metrics["qwk"],
        "critical_under_triage_rate": clinical_metrics["critical_under_triage_rate"],
        "ur_95_ci_str": clinical_metrics["ur_ci_str"],
        "gpu_p95_latency_ms": lat_gpu["p95_latency_ms"],
        "gpu_qps": lat_gpu["qps"],
        "cpu_p95_latency_ms": lat_cpu["p95_latency_ms"],
        "cpu_qps": lat_cpu["qps"]
    }
    
    with open(os.path.join(output_dir, "task4_e2e_results", "task4_e2e_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(e2e_results, f, indent=2)
        
    if notifier:
        notifier.notify_step_complete(
            task_name="Task 4: End-to-End Pipeline",
            step_name="Full Integrated Pipeline Evaluation",
            metrics={
                "Overall E2E F1": e2e_results["overall_e2e_f1"],
                "Overall E2E F2": e2e_results["overall_e2e_f2"],
                "Clinical QWK": e2e_results["clinical_qwk"],
                "Under-triage Rate": e2e_results["critical_under_triage_rate"],
                "CPU P95 Latency (ms)": e2e_results["cpu_p95_latency_ms"]
            }
        )
        notifier.notify_task_complete(
            task_name="Task 4: End-to-End Integrated Pipeline",
            summary_info=f"E2E System Benchmark Complete.\n📊 **E2E F1**: `{e2e_results['overall_e2e_f1']:.4f}` | **Under-triage Rate**: `{e2e_results['critical_under_triage_rate']:.2f}%`"
        )
        
    return e2e_results

