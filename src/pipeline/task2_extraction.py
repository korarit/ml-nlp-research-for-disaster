"""
Task 2: NER & Entity/Count Extraction Pipeline.
Evaluates Approach A (Rules), Approach B1 (Binned Classifiers), Approach B2 (Continuous Regressors),
Approach B3 (CRF Tagger), and Approach C (Hybrid System) on Gemini CV and Luna Held-out Test.
"""

import os
import gc
import json
import pickle
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import scipy.sparse as sp
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.utils.data_loader import load_all_datasets, bin_count_target, COUNT_COLUMNS, DEFAULT_TRAIN_PATH, DEFAULT_TEST_PATH
from src.features.text_vectorizer import create_tfidf_vectorizer
from src.models.classifiers import get_classifier, ALL_CLASSIFIER_NAMES
from src.models.regressors import get_regressor, ALL_REGRESSOR_NAMES
from src.models.rules_engine import ExtractionRulesEngine
from src.models.ner_crf import (
    ThaiLocationCRFTagger, ThaiMultiNER_CRFTagger,
    SlidingWindowTokenClassifier, BiLSTM_CRF_Tagger,
    Standard_LSTM_Tagger,
    prepare_task2_token_cache
)
from src.utils.metrics import (
    compute_count_regression_metrics, compute_string_match_metrics,
    compute_classification_metrics
)
from src.utils.statistical_tests import run_pairwise_model_stat_tests
from src.utils.visualization import plot_02_model_performance_comparison, plot_task2_extraction_comparison


def extract_gt_entity_vector(df: pd.DataFrame, col: str) -> List[str]:
    """Extracts entity values from dataframe, replacing NaNs/nulls with empty string."""
    return [str(val).strip() if pd.notna(val) and str(val).lower() not in ("nan", "none", "null") else "" for val in df.get(col, [])]


def extract_gt_coords_vector(df: pd.DataFrame) -> List[str]:
    """Extracts coordinates string lat,lng from dataframe, filtering 0.0 lat/lng as empty."""
    coords = []
    for _, row in df.iterrows():
        lat = row.get("gt_lat")
        lng = row.get("gt_lng")
        if pd.notna(lat) and float(lat) != 0.0:
            coords.append(f"{lat},{lng}")
        else:
            coords.append("")
    return coords


def run_task2_approach_a_rules(test_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluates Approach A Rule-Based Extraction Engine on Test Dataset across all entities and counts."""
    engine = ExtractionRulesEngine()
    texts = test_df["generated_text"].tolist()
    
    # Phone match
    true_phones = extract_gt_entity_vector(test_df, "gt_victim_phone")
    pred_phones = [str(engine.extract_phone(t) or "") for t in texts]
    phone_m = compute_string_match_metrics(true_phones, pred_phones)
    
    # Map URL match
    true_urls = extract_gt_entity_vector(test_df, "gt_google_map_url")
    pred_urls = [str(engine.extract_map_url(t) or "") for t in texts]
    url_m = compute_string_match_metrics(true_urls, pred_urls)
    
    # Location match
    true_locs = extract_gt_entity_vector(test_df, "gt_location_name")
    pred_locs = [str(engine.extract_location(t) or "") for t in texts]
    loc_m = compute_string_match_metrics(true_locs, pred_locs)
    
    # Coordinates match
    true_coord_strs = extract_gt_coords_vector(test_df)
    pred_coords = [engine.extract_coords(t) for t in texts]
    pred_coord_strs = [f"{c[0]},{c[1]}" if c[0] is not None else "" for c in pred_coords]
    coords_m = compute_string_match_metrics(true_coord_strs, pred_coord_strs)

    # Names match
    true_vics = extract_gt_entity_vector(test_df, "gt_victim_name")
    pred_vics = [str(engine.extract_name(t) or "") for t in texts]
    vic_m = compute_string_match_metrics(true_vics, pred_vics)

    true_reps = extract_gt_entity_vector(test_df, "gt_reporter_name")
    pred_reps = [str(engine.extract_name(t) or "") for t in texts]
    rep_m = compute_string_match_metrics(true_reps, pred_reps)
    
    # Count metrics
    count_maes = []
    count_rmses = []
    count_ems = []
    count_f1s = []
    count_f2s = []
    nonzero_maes = []
    nonzero_ems = []
    for field in COUNT_COLUMNS:
        if field in test_df.columns:
            y_t = test_df[field].values.astype(float)
            y_p = np.array([engine.extract_count(t, field) for t in texts], dtype=float)
            m = compute_count_regression_metrics(y_t, y_p)
            count_maes.append(m["mae"])
            count_rmses.append(m["rmse"])
            count_ems.append(m["exact_match"])
            nonzero_maes.append(m["nonzero_mae"])
            nonzero_ems.append(m["nonzero_exact_match"])
            
            clf_m = compute_classification_metrics(bin_count_target(y_t), bin_count_target(y_p))
            count_f1s.append(clf_m["f1_weighted"])
            count_f2s.append(clf_m["f2"])
            
    return {
        "approach": "Approach A (Rules)",
        "f1": float(np.mean(count_f1s)) if count_f1s else 0.0,
        "f2": float(np.mean(count_f2s)) if count_f2s else 0.0,
        "victim_name_exact_match": vic_m["gt_match_rate"],
        "victim_name_f1": vic_m["f1"],
        "reporter_name_exact_match": rep_m["gt_match_rate"],
        "reporter_name_f1": rep_m["f1"],
        "phone_exact_match": phone_m["gt_match_rate"],
        "phone_f1": phone_m["f1"],
        "map_url_exact_match": url_m["gt_match_rate"],
        "map_url_f1": url_m["f1"],
        "coords_exact_match": coords_m["gt_match_rate"],
        "coords_f1": coords_m["f1"],
        "location_exact_match": loc_m["gt_match_rate"],
        "location_f1": loc_m["f1"],
        "mean_count_mae": float(np.mean(count_maes)) if count_maes else 0.0,
        "mean_count_rmse": float(np.mean(count_rmses)) if count_rmses else 0.0,
        "count_exact_match": float(np.mean(count_ems)) if count_ems else 0.0,
        "nonzero_count_mae": float(np.mean(nonzero_maes)) if nonzero_maes else 0.0,
        "nonzero_count_exact_match": float(np.mean(nonzero_ems)) if nonzero_ems else 0.0
    }


def run_task2_approach_b1_binned(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "LogisticRegression",
    use_gpu: bool = True,
    X_train_vec: Optional[Any] = None,
    X_test_vec: Optional[Any] = None,
    lstm_tagger: Optional[Standard_LSTM_Tagger] = None,
    lstm_token_cache: Optional[Dict[str, Any]] = None,
    token_cache: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """
    Evaluates Approach B1: LSTM Sequence Extractor -> Classical ML Classifiers (17 Models) for Counts and Entity Tokens.
    """
    if X_train_vec is None or X_test_vec is None:
        vectorizer = create_tfidf_vectorizer(use_hybrid=True)
        X_train_vec = vectorizer.fit_transform(train_df["generated_text"].values)
        X_test_vec = vectorizer.transform(test_df["generated_text"].values)
    
    # 1. Classical Classifier for Counts
    maes = []
    rmses = []
    ems = []
    f1s = []
    f2s = []
    nonzero_maes = []
    nonzero_ems = []
    for field in COUNT_COLUMNS:
        if field in train_df.columns and field in test_df.columns:
            y_tr_raw = train_df[field].values.astype(float)
            y_te_raw = test_df[field].values.astype(float)
            y_tr_binned = bin_count_target(y_tr_raw)
            y_te_binned = bin_count_target(y_te_raw)
            
            if len(np.unique(y_tr_binned)) < 2:
                preds_binned = np.full(len(y_te_binned), y_tr_binned[0] if len(y_tr_binned) > 0 else 0)
            else:
                clf = get_classifier(model_name, use_gpu=use_gpu)
                clf.fit(X_train_vec, y_tr_binned)
                preds_binned = clf.predict(X_test_vec)
            
            m = compute_count_regression_metrics(y_te_binned, preds_binned)
            clf_m = compute_classification_metrics(y_te_binned, preds_binned)
            maes.append(m["mae"])
            rmses.append(m["rmse"])
            ems.append(m["exact_match"])
            nonzero_maes.append(m["nonzero_mae"])
            nonzero_ems.append(m["nonzero_exact_match"])
            f1s.append(clf_m["f1_weighted"])
            f2s.append(clf_m["f2"])
            
    # 2. Classical Classifier deciding on LSTM Token Embeddings for Entity Recognition
    if lstm_tagger is not None and lstm_token_cache is not None and len(lstm_token_cache.get("X_train", [])) > 0:
        clf_token = get_classifier(model_name, use_gpu=use_gpu)
        clf_token.fit(lstm_token_cache["X_train"], lstm_token_cache["y_train"])
        preds_entities = lstm_tagger.predict_entities_from_classifier(clf_token, lstm_token_cache)
    elif lstm_tagger is not None:
        preds_entities = lstm_tagger.predict_entities(test_df["generated_text"].tolist())
    else:
        lstm_tagger = Standard_LSTM_Tagger(use_gpu=use_gpu)
        lstm_tagger.fit(
            train_df["generated_text"].tolist(),
            extract_gt_entity_vector(train_df, "gt_location_name"),
            extract_gt_entity_vector(train_df, "gt_victim_phone"),
            extract_gt_entity_vector(train_df, "gt_google_map_url"),
            train_df.get("gt_lat", [None] * len(train_df)).tolist(),
            train_df.get("gt_lng", [None] * len(train_df)).tolist(),
            extract_gt_entity_vector(train_df, "gt_victim_name"),
            extract_gt_entity_vector(train_df, "gt_reporter_name")
        )
        preds_entities = lstm_tagger.predict_entities(test_df["generated_text"].tolist())
    
    loc_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_location_name"), preds_entities["locations"])
    phone_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_victim_phone"), preds_entities["phones"])
    url_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_google_map_url"), preds_entities["urls"])
    coords_m = compute_string_match_metrics(extract_gt_coords_vector(test_df), preds_entities["coords"])
    vic_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_victim_name"), preds_entities["victim_names"])
    rep_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_reporter_name"), preds_entities["reporter_names"])
    
    return {
        "approach": f"Approach B1 (Binned {model_name})",
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "f2": float(np.mean(f2s)) if f2s else 0.0,
        "victim_name_exact_match": vic_m["gt_match_rate"],
        "victim_name_f1": vic_m["f1"],
        "reporter_name_exact_match": rep_m["gt_match_rate"],
        "reporter_name_f1": rep_m["f1"],
        "phone_exact_match": phone_m["gt_match_rate"],
        "phone_f1": phone_m["f1"],
        "map_url_exact_match": url_m["gt_match_rate"],
        "map_url_f1": url_m["f1"],
        "coords_exact_match": coords_m["gt_match_rate"],
        "coords_f1": coords_m["f1"],
        "location_exact_match": loc_m["gt_match_rate"],
        "location_f1": loc_m["f1"],
        "mean_count_mae": float(np.mean(maes)) if maes else 0.0,
        "mean_count_rmse": float(np.mean(rmses)) if rmses else 0.0,
        "count_exact_match": float(np.mean(ems)) if ems else 0.0,
        "nonzero_count_mae": float(np.mean(nonzero_maes)) if nonzero_maes else 0.0,
        "nonzero_count_exact_match": float(np.mean(nonzero_ems)) if nonzero_ems else 0.0
    }, preds_entities


def run_task2_approach_b2_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "XGBRegressor",
    use_gpu: bool = True,
    X_train_vec: Optional[Any] = None,
    X_test_vec: Optional[Any] = None,
    lstm_tagger: Optional[Standard_LSTM_Tagger] = None,
    lstm_token_cache: Optional[Dict[str, Any]] = None,
    token_cache: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, Dict[str, List[str]]]:
    """
    Evaluates Approach B2: LSTM Sequence Extractor -> Classical ML Regressors (17 Models) for Counts and Entity Tokens.
    """
    if X_train_vec is None or X_test_vec is None:
        vectorizer = create_tfidf_vectorizer(use_hybrid=True)
        X_train_vec = vectorizer.fit_transform(train_df["generated_text"].values)
        X_test_vec = vectorizer.transform(test_df["generated_text"].values)
    
    # 1. Classical Regressors for Counts
    maes = []
    rmses = []
    ems = []
    f1s = []
    f2s = []
    nonzero_maes = []
    nonzero_ems = []
    all_y_true = []
    all_y_pred = []
    
    for field in COUNT_COLUMNS:
        if field in train_df.columns and field in test_df.columns:
            y_tr = train_df[field].values.astype(float)
            y_te = test_df[field].values.astype(float)
            
            reg = get_regressor(model_name, use_gpu=use_gpu)
            reg.fit(X_train_vec, y_tr)
            preds = np.clip(reg.predict(X_test_vec), 0, None)
            
            m = compute_count_regression_metrics(y_te, preds)
            clf_m = compute_classification_metrics(bin_count_target(y_te), bin_count_target(preds))
            maes.append(m["mae"])
            rmses.append(m["rmse"])
            ems.append(m["exact_match"])
            nonzero_maes.append(m["nonzero_mae"])
            nonzero_ems.append(m["nonzero_exact_match"])
            f1s.append(clf_m["f1_weighted"])
            f2s.append(clf_m["f2"])
            all_y_true.extend(y_te)
            all_y_pred.extend(preds)

    # 2. Classical Classifier deciding on LSTM Token Embeddings for Entity Recognition
    paired_clf = model_name.replace("Regressor", "Classifier")
    if paired_clf not in ALL_CLASSIFIER_NAMES:
        paired_clf = "LogisticRegression"
        
    if lstm_tagger is not None and lstm_token_cache is not None and len(lstm_token_cache.get("X_train", [])) > 0:
        clf_token = get_classifier(paired_clf, use_gpu=use_gpu)
        clf_token.fit(lstm_token_cache["X_train"], lstm_token_cache["y_train"])
        preds_entities = lstm_tagger.predict_entities_from_classifier(clf_token, lstm_token_cache)
    elif lstm_tagger is not None:
        preds_entities = lstm_tagger.predict_entities(test_df["generated_text"].tolist())
    else:
        lstm_tagger = Standard_LSTM_Tagger(use_gpu=use_gpu)
        lstm_tagger.fit(
            train_df["generated_text"].tolist(),
            extract_gt_entity_vector(train_df, "gt_location_name"),
            extract_gt_entity_vector(train_df, "gt_victim_phone"),
            extract_gt_entity_vector(train_df, "gt_google_map_url"),
            train_df.get("gt_lat", [None] * len(train_df)).tolist(),
            train_df.get("gt_lng", [None] * len(train_df)).tolist(),
            extract_gt_entity_vector(train_df, "gt_victim_name"),
            extract_gt_entity_vector(train_df, "gt_reporter_name")
        )
        preds_entities = lstm_tagger.predict_entities(test_df["generated_text"].tolist())
    
    loc_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_location_name"), preds_entities["locations"])
    phone_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_victim_phone"), preds_entities["phones"])
    url_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_google_map_url"), preds_entities["urls"])
    coords_m = compute_string_match_metrics(extract_gt_coords_vector(test_df), preds_entities["coords"])
    vic_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_victim_name"), preds_entities["victim_names"])
    rep_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_reporter_name"), preds_entities["reporter_names"])
            
    return {
        "approach": f"Approach B2 (Regressor {model_name})",
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "f2": float(np.mean(f2s)) if f2s else 0.0,
        "victim_name_exact_match": vic_m["gt_match_rate"],
        "victim_name_f1": vic_m["f1"],
        "reporter_name_exact_match": rep_m["gt_match_rate"],
        "reporter_name_f1": rep_m["f1"],
        "phone_exact_match": phone_m["gt_match_rate"],
        "phone_f1": phone_m["f1"],
        "map_url_exact_match": url_m["gt_match_rate"],
        "map_url_f1": url_m["f1"],
        "coords_exact_match": coords_m["gt_match_rate"],
        "coords_f1": coords_m["f1"],
        "location_exact_match": loc_m["gt_match_rate"],
        "location_f1": loc_m["f1"],
        "mean_count_mae": float(np.mean(maes)) if maes else 0.0,
        "mean_count_rmse": float(np.mean(rmses)) if rmses else 0.0,
        "count_exact_match": float(np.mean(ems)) if ems else 0.0,
        "nonzero_count_mae": float(np.mean(nonzero_maes)) if nonzero_maes else 0.0,
        "nonzero_count_exact_match": float(np.mean(nonzero_ems)) if nonzero_ems else 0.0
    }, np.array(all_y_true), np.array(all_y_pred), preds_entities


def run_task2_approach_b3_binned_bilstm_crf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "LogisticRegression",
    use_gpu: bool = True,
    X_train_vec: Optional[Any] = None,
    X_test_vec: Optional[Any] = None,
    bilstm_tagger: Optional[BiLSTM_CRF_Tagger] = None,
    bilstm_token_cache: Optional[Dict[str, Any]] = None,
    token_cache: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """
    Evaluates Approach B3: BiLSTM-CRF Sequence Extractor -> Classical ML Classifiers (17 Models) for Counts and Entity Tokens.
    """
    if X_train_vec is None or X_test_vec is None:
        vectorizer = create_tfidf_vectorizer(use_hybrid=True)
        X_train_vec = vectorizer.fit_transform(train_df["generated_text"].values)
        X_test_vec = vectorizer.transform(test_df["generated_text"].values)
    
    # 1. Classical Classifier for Counts
    maes, rmses, ems, f1s, f2s, nonzero_maes, nonzero_ems = [], [], [], [], [], [], []
    for field in COUNT_COLUMNS:
        if field in train_df.columns and field in test_df.columns:
            y_tr_raw = train_df[field].values.astype(float)
            y_te_raw = test_df[field].values.astype(float)
            y_tr_binned = bin_count_target(y_tr_raw)
            y_te_binned = bin_count_target(y_te_raw)
            
            if len(np.unique(y_tr_binned)) < 2:
                preds_binned = np.full(len(y_te_binned), y_tr_binned[0] if len(y_tr_binned) > 0 else 0)
            else:
                clf = get_classifier(model_name, use_gpu=use_gpu)
                clf.fit(X_train_vec, y_tr_binned)
                preds_binned = clf.predict(X_test_vec)
            
            m = compute_count_regression_metrics(y_te_binned, preds_binned)
            clf_m = compute_classification_metrics(y_te_binned, preds_binned)
            maes.append(m["mae"])
            rmses.append(m["rmse"])
            ems.append(m["exact_match"])
            nonzero_maes.append(m["nonzero_mae"])
            nonzero_ems.append(m["nonzero_exact_match"])
            f1s.append(clf_m["f1_weighted"])
            f2s.append(clf_m["f2"])
            
    # 2. Classical Classifier deciding on BiLSTM Token Embeddings for Entity Recognition
    if bilstm_tagger is not None and bilstm_token_cache is not None and len(bilstm_token_cache.get("X_train", [])) > 0:
        clf_token = get_classifier(model_name, use_gpu=use_gpu)
        clf_token.fit(bilstm_token_cache["X_train"], bilstm_token_cache["y_train"])
        preds_entities = bilstm_tagger.predict_entities_from_classifier(clf_token, bilstm_token_cache)
    elif bilstm_tagger is not None:
        preds_entities = bilstm_tagger.predict_entities(test_df["generated_text"].tolist())
    else:
        bilstm_tagger = BiLSTM_CRF_Tagger(use_gpu=use_gpu)
        bilstm_tagger.fit(
            train_df["generated_text"].tolist(),
            extract_gt_entity_vector(train_df, "gt_location_name"),
            extract_gt_entity_vector(train_df, "gt_victim_phone"),
            extract_gt_entity_vector(train_df, "gt_google_map_url"),
            train_df.get("gt_lat", [None] * len(train_df)).tolist(),
            train_df.get("gt_lng", [None] * len(train_df)).tolist(),
            extract_gt_entity_vector(train_df, "gt_victim_name"),
            extract_gt_entity_vector(train_df, "gt_reporter_name")
        )
        preds_entities = bilstm_tagger.predict_entities(test_df["generated_text"].tolist())
    
    loc_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_location_name"), preds_entities["locations"])
    phone_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_victim_phone"), preds_entities["phones"])
    url_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_google_map_url"), preds_entities["urls"])
    coords_m = compute_string_match_metrics(extract_gt_coords_vector(test_df), preds_entities["coords"])
    vic_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_victim_name"), preds_entities["victim_names"])
    rep_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_reporter_name"), preds_entities["reporter_names"])
    
    return {
        "approach": f"Approach B3 (Binned {model_name} + BiLSTM-CRF)",
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "f2": float(np.mean(f2s)) if f2s else 0.0,
        "victim_name_exact_match": vic_m["gt_match_rate"],
        "victim_name_f1": vic_m["f1"],
        "reporter_name_exact_match": rep_m["gt_match_rate"],
        "reporter_name_f1": rep_m["f1"],
        "phone_exact_match": phone_m["gt_match_rate"],
        "phone_f1": phone_m["f1"],
        "map_url_exact_match": url_m["gt_match_rate"],
        "map_url_f1": url_m["f1"],
        "coords_exact_match": coords_m["gt_match_rate"],
        "coords_f1": coords_m["f1"],
        "location_exact_match": loc_m["gt_match_rate"],
        "location_f1": loc_m["f1"],
        "mean_count_mae": float(np.mean(maes)) if maes else 0.0,
        "mean_count_rmse": float(np.mean(rmses)) if rmses else 0.0,
        "count_exact_match": float(np.mean(ems)) if ems else 0.0,
        "nonzero_count_mae": float(np.mean(nonzero_maes)) if nonzero_maes else 0.0,
        "nonzero_count_exact_match": float(np.mean(nonzero_ems)) if nonzero_ems else 0.0
    }, preds_entities


def run_task2_approach_b4_regression_bilstm_crf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "XGBRegressor",
    use_gpu: bool = True,
    X_train_vec: Optional[Any] = None,
    X_test_vec: Optional[Any] = None,
    bilstm_tagger: Optional[BiLSTM_CRF_Tagger] = None,
    bilstm_token_cache: Optional[Dict[str, Any]] = None,
    token_cache: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, Dict[str, List[str]]]:
    """
    Evaluates Approach B4: BiLSTM-CRF Sequence Extractor -> Classical ML Regressors (17 Models) for Counts and Entity Tokens.
    """
    if X_train_vec is None or X_test_vec is None:
        vectorizer = create_tfidf_vectorizer(use_hybrid=True)
        X_train_vec = vectorizer.fit_transform(train_df["generated_text"].values)
        X_test_vec = vectorizer.transform(test_df["generated_text"].values)
    
    # 1. Classical Regressors for Counts
    maes, rmses, ems, f1s, f2s, nonzero_maes, nonzero_ems = [], [], [], [], [], [], []
    all_y_true = []
    all_y_pred = []
    
    for field in COUNT_COLUMNS:
        if field in train_df.columns and field in test_df.columns:
            y_tr = train_df[field].values.astype(float)
            y_te = test_df[field].values.astype(float)
            
            reg = get_regressor(model_name, use_gpu=use_gpu)
            reg.fit(X_train_vec, y_tr)
            preds = np.clip(reg.predict(X_test_vec), 0, None)
            
            m = compute_count_regression_metrics(y_te, preds)
            clf_m = compute_classification_metrics(bin_count_target(y_te), bin_count_target(preds))
            maes.append(m["mae"])
            rmses.append(m["rmse"])
            ems.append(m["exact_match"])
            nonzero_maes.append(m["nonzero_mae"])
            nonzero_ems.append(m["nonzero_exact_match"])
            f1s.append(clf_m["f1_weighted"])
            f2s.append(clf_m["f2"])
            all_y_true.extend(y_te)
            all_y_pred.extend(preds)

    # 2. Classical Classifier deciding on BiLSTM Token Embeddings for Entity Recognition
    paired_clf = model_name.replace("Regressor", "Classifier")
    if paired_clf not in ALL_CLASSIFIER_NAMES:
        paired_clf = "LogisticRegression"
        
    if bilstm_tagger is not None and bilstm_token_cache is not None and len(bilstm_token_cache.get("X_train", [])) > 0:
        clf_token = get_classifier(paired_clf, use_gpu=use_gpu)
        clf_token.fit(bilstm_token_cache["X_train"], bilstm_token_cache["y_train"])
        preds_entities = bilstm_tagger.predict_entities_from_classifier(clf_token, bilstm_token_cache)
    elif bilstm_tagger is not None:
        preds_entities = bilstm_tagger.predict_entities(test_df["generated_text"].tolist())
    else:
        bilstm_tagger = BiLSTM_CRF_Tagger(use_gpu=use_gpu)
        bilstm_tagger.fit(
            train_df["generated_text"].tolist(),
            extract_gt_entity_vector(train_df, "gt_location_name"),
            extract_gt_entity_vector(train_df, "gt_victim_phone"),
            extract_gt_entity_vector(train_df, "gt_google_map_url"),
            train_df.get("gt_lat", [None] * len(train_df)).tolist(),
            train_df.get("gt_lng", [None] * len(train_df)).tolist(),
            extract_gt_entity_vector(train_df, "gt_victim_name"),
            extract_gt_entity_vector(train_df, "gt_reporter_name")
        )
        preds_entities = bilstm_tagger.predict_entities(test_df["generated_text"].tolist())
    
    loc_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_location_name"), preds_entities["locations"])
    phone_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_victim_phone"), preds_entities["phones"])
    url_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_google_map_url"), preds_entities["urls"])
    coords_m = compute_string_match_metrics(extract_gt_coords_vector(test_df), preds_entities["coords"])
    vic_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_victim_name"), preds_entities["victim_names"])
    rep_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_reporter_name"), preds_entities["reporter_names"])
            
    return {
        "approach": f"Approach B4 (Regressor {model_name} + BiLSTM-CRF)",
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "f2": float(np.mean(f2s)) if f2s else 0.0,
        "victim_name_exact_match": vic_m["gt_match_rate"],
        "victim_name_f1": vic_m["f1"],
        "reporter_name_exact_match": rep_m["gt_match_rate"],
        "reporter_name_f1": rep_m["f1"],
        "phone_exact_match": phone_m["gt_match_rate"],
        "phone_f1": phone_m["f1"],
        "map_url_exact_match": url_m["gt_match_rate"],
        "map_url_f1": url_m["f1"],
        "coords_exact_match": coords_m["gt_match_rate"],
        "coords_f1": coords_m["f1"],
        "location_exact_match": loc_m["gt_match_rate"],
        "location_f1": loc_m["f1"],
        "mean_count_mae": float(np.mean(maes)) if maes else 0.0,
        "mean_count_rmse": float(np.mean(rmses)) if rmses else 0.0,
        "count_exact_match": float(np.mean(ems)) if ems else 0.0,
        "nonzero_count_mae": float(np.mean(nonzero_maes)) if nonzero_maes else 0.0,
        "nonzero_count_exact_match": float(np.mean(nonzero_ems)) if nonzero_ems else 0.0
    }, np.array(all_y_true), np.array(all_y_pred), preds_entities


def run_task2_approach_c_hybrid(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_regressor_name: str = "XGBRegressor",
    best_token_model_name: str = "CRF",
    use_gpu: bool = True,
    X_train_vec: Optional[Any] = None,
    X_test_vec: Optional[Any] = None,
    best_pred_entities: Optional[Dict[str, List[str]]] = None,
    token_cache: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluates Approach C: Adaptive Hybrid System
    (Best ML Token Tagger for Names & Location + Best ML Regressor for counts + Regex Rules Engine for Phone/URL/Coords).
    """
    res_a = run_task2_approach_a_rules(test_df)
    res_reg, _, _, _ = run_task2_approach_b2_regression(
        train_df, test_df, best_regressor_name, use_gpu=use_gpu,
        X_train_vec=X_train_vec, X_test_vec=X_test_vec,
        token_cache=token_cache
    )
    
    if best_pred_entities is not None:
        loc_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_location_name"), best_pred_entities["locations"])
        vic_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_victim_name"), best_pred_entities["victim_names"])
        rep_m = compute_string_match_metrics(extract_gt_entity_vector(test_df, "gt_reporter_name"), best_pred_entities["reporter_names"])
        loc_exact, loc_f1 = loc_m["gt_match_rate"], loc_m["f1"]
        vic_exact, vic_f1 = vic_m["gt_match_rate"], vic_m["f1"]
        rep_exact, rep_f1 = rep_m["gt_match_rate"], rep_m["f1"]
    else:
        loc_exact, loc_f1 = res_a["location_exact_match"], res_a["location_f1"]
        vic_exact, vic_f1 = res_a["victim_name_exact_match"], res_a["victim_name_f1"]
        rep_exact, rep_f1 = res_a["reporter_name_exact_match"], res_a["reporter_name_f1"]
    
    return {
        "approach": f"Approach C (Hybrid Rules + ML {best_token_model_name} + {best_regressor_name}) [Best Hybrid]",
        "f1": res_reg["f1"],
        "f2": res_reg["f2"],
        "victim_name_exact_match": vic_exact,
        "victim_name_f1": vic_f1,
        "reporter_name_exact_match": rep_exact,
        "reporter_name_f1": rep_f1,
        "phone_exact_match": res_a["phone_exact_match"],
        "phone_f1": res_a["phone_f1"],
        "map_url_exact_match": res_a["map_url_exact_match"],
        "map_url_f1": res_a["map_url_f1"],
        "coords_exact_match": res_a["coords_exact_match"],
        "coords_f1": res_a["coords_f1"],
        "location_exact_match": loc_exact,
        "location_f1": loc_f1,
        "mean_count_mae": res_reg["mean_count_mae"],
        "mean_count_rmse": res_reg["mean_count_rmse"],
        "count_exact_match": res_reg["count_exact_match"],
        "nonzero_count_mae": res_reg["nonzero_count_mae"],
        "nonzero_count_exact_match": res_reg["nonzero_count_exact_match"]
    }


def execute_task2_pipeline(
    output_dir: str,
    selected_models: List[str] = None,
    use_gpu: bool = True,
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
    
    # Pre-compute TF-IDF Vectorizer once for all ML Approaches
    print("--- [Task 2] Pre-computing TF-IDF Vectorization (Thai tokenization) once for pipeline ---")
    vectorizer = create_tfidf_vectorizer(use_hybrid=True)
    X_train_vec = vectorizer.fit_transform(train_df["generated_text"].values)
    X_test_vec = vectorizer.transform(test_df["generated_text"].values)
    print(f"--- [Task 2] TF-IDF Matrix ready: train={X_train_vec.shape}, test={X_test_vec.shape} ---")

    # Pre-compute Token Feature Matrix once for all ML Token Classifiers
    print("--- [Task 2] Pre-computing Token Matrices once for all Token Classifiers ---")
    token_cache = prepare_task2_token_cache(train_df, test_df)
    print(f"--- [Task 2] Token Matrix ready: train={token_cache['X_train_mat'].shape}, test={token_cache['X_test_mat'].shape} ---")

    # Models & Feature Checkpoints directory
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Pre-train / Load Standard LSTM once for Entity Extraction across Approaches B1, B2
    lstm_model_path = os.path.join(models_dir, "standard_lstm_tagger.pt")
    lstm_cache_path = os.path.join(models_dir, "standard_lstm_features.pkl")
    
    if os.path.exists(lstm_model_path) and os.path.exists(lstm_cache_path) and not force:
        print(f"--- [Task 2] Loading cached Standard LSTM model & features from {models_dir} (Skipping training) ---")
        lstm_tagger = Standard_LSTM_Tagger.load(lstm_model_path, use_gpu=use_gpu)
        with open(lstm_cache_path, "rb") as f:
            lstm_cache_data = pickle.load(f)
            X_train_lstm_combined = lstm_cache_data["X_train_lstm_combined"]
            X_test_lstm_combined = lstm_cache_data["X_test_lstm_combined"]
            lstm_token_cache = lstm_cache_data["lstm_token_cache"]
    else:
        print("--- [Task 2] Training Standard LSTM Sequence Tagger on GPU/CPU ---")
        lstm_tagger = Standard_LSTM_Tagger(use_gpu=use_gpu)
        lstm_tagger.fit(
            train_df["generated_text"].tolist(),
            extract_gt_entity_vector(train_df, "gt_location_name"),
            extract_gt_entity_vector(train_df, "gt_victim_phone"),
            extract_gt_entity_vector(train_df, "gt_google_map_url"),
            train_df.get("gt_lat", [None] * len(train_df)).tolist(),
            train_df.get("gt_lng", [None] * len(train_df)).tolist(),
            extract_gt_entity_vector(train_df, "gt_victim_name"),
            extract_gt_entity_vector(train_df, "gt_reporter_name")
        )
        X_train_lstm_sent = lstm_tagger.extract_sentence_embeddings(train_df["generated_text"].tolist())
        X_test_lstm_sent = lstm_tagger.extract_sentence_embeddings(test_df["generated_text"].tolist())
        X_train_lstm_combined = sp.hstack([X_train_vec, X_train_lstm_sent], format="csr")
        X_test_lstm_combined = sp.hstack([X_test_vec, X_test_lstm_sent], format="csr")
        lstm_token_cache = lstm_tagger.prepare_token_feature_cache(train_df, test_df)
        
        lstm_tagger.save(lstm_model_path)
        with open(lstm_cache_path, "wb") as f:
            pickle.dump({
                "X_train_lstm_combined": X_train_lstm_combined,
                "X_test_lstm_combined": X_test_lstm_combined,
                "lstm_token_cache": lstm_token_cache
            }, f)
        print(f"--- [Task 2] Standard LSTM model & features saved to {models_dir} ---")

    # Pre-train / Load BiLSTM-CRF once for Entity Extraction across Approaches B3, B4
    bilstm_model_path = os.path.join(models_dir, "bilstm_crf_tagger.pt")
    bilstm_cache_path = os.path.join(models_dir, "bilstm_crf_features.pkl")
    
    if os.path.exists(bilstm_model_path) and os.path.exists(bilstm_cache_path) and not force:
        print(f"--- [Task 2] Loading cached BiLSTM-CRF model & features from {models_dir} (Skipping training) ---")
        bilstm_tagger = BiLSTM_CRF_Tagger.load(bilstm_model_path, use_gpu=use_gpu)
        with open(bilstm_cache_path, "rb") as f:
            bilstm_cache_data = pickle.load(f)
            X_train_bilstm_combined = bilstm_cache_data["X_train_bilstm_combined"]
            X_test_bilstm_combined = bilstm_cache_data["X_test_bilstm_combined"]
            bilstm_token_cache = bilstm_cache_data["bilstm_token_cache"]
    else:
        print("--- [Task 2] Training BiLSTM-CRF Sequence Tagger on GPU/CPU ---")
        bilstm_tagger = BiLSTM_CRF_Tagger(use_gpu=use_gpu)
        bilstm_tagger.fit(
            train_df["generated_text"].tolist(),
            extract_gt_entity_vector(train_df, "gt_location_name"),
            extract_gt_entity_vector(train_df, "gt_victim_phone"),
            extract_gt_entity_vector(train_df, "gt_google_map_url"),
            train_df.get("gt_lat", [None] * len(train_df)).tolist(),
            train_df.get("gt_lng", [None] * len(train_df)).tolist(),
            extract_gt_entity_vector(train_df, "gt_victim_name"),
            extract_gt_entity_vector(train_df, "gt_reporter_name")
        )
        X_train_bilstm_sent = bilstm_tagger.extract_sentence_embeddings(train_df["generated_text"].tolist())
        X_test_bilstm_sent = bilstm_tagger.extract_sentence_embeddings(test_df["generated_text"].tolist())
        X_train_bilstm_combined = sp.hstack([X_train_vec, X_train_bilstm_sent], format="csr")
        X_test_bilstm_combined = sp.hstack([X_test_vec, X_test_bilstm_sent], format="csr")
        bilstm_token_cache = bilstm_tagger.prepare_token_feature_cache(train_df, test_df)
        
        bilstm_tagger.save(bilstm_model_path)
        with open(bilstm_cache_path, "wb") as f:
            pickle.dump({
                "X_train_bilstm_combined": X_train_bilstm_combined,
                "X_test_bilstm_combined": X_test_bilstm_combined,
                "bilstm_token_cache": bilstm_token_cache
            }, f)
        print(f"--- [Task 2] BiLSTM-CRF model & features saved to {models_dir} ---")

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
        print(f"--- Task 2 Running: {app_a_name} ---")
        res_a = run_task2_approach_a_rules(test_df)
        results.append(res_a)
        pd.DataFrame(results).to_csv(task2_csv, index=False)
        notify_step(res_a)
    
    # Model lists
    if selected_models:
        clf_models = []
        for m in selected_models:
            if m in ALL_CLASSIFIER_NAMES or m == "DummyClassifier":
                clf_models.append(m)
            elif f"{m}Classifier" in ALL_CLASSIFIER_NAMES:
                clf_models.append(f"{m}Classifier")
            elif m.replace("Regressor", "Classifier") in ALL_CLASSIFIER_NAMES:
                clf_models.append(m.replace("Regressor", "Classifier"))
            else:
                clf_models.append(m)
                
        reg_models = []
        for m in selected_models:
            if m in ALL_REGRESSOR_NAMES or m == "DummyRegressor":
                reg_models.append(m)
            elif f"{m}Regressor" in ALL_REGRESSOR_NAMES:
                reg_models.append(f"{m}Regressor")
            elif m.replace("Classifier", "Regressor") in ALL_REGRESSOR_NAMES:
                reg_models.append(m.replace("Classifier", "Regressor"))
            else:
                reg_models.append(m)
    else:
        clf_models = ALL_CLASSIFIER_NAMES
        reg_models = ALL_REGRESSOR_NAMES

    # 2. Approach B1 (Binned Classifiers with Standard LSTM Feature Extractor)
    for clf_name in clf_models:
        app_b1_name = f"Approach B1 (Binned {clf_name})"
        if not should_skip(app_b1_name):
            print(f"--- Task 2 Running: {app_b1_name} ---")
            try:
                res_b1, _ = run_task2_approach_b1_binned(
                    train_df, test_df, clf_name,
                    use_gpu=use_gpu, X_train_vec=X_train_lstm_combined, X_test_vec=X_test_lstm_combined,
                    lstm_tagger=lstm_tagger,
                    lstm_token_cache=lstm_token_cache,
                    token_cache=token_cache
                )
                results.append(res_b1)
                pd.DataFrame(results).to_csv(task2_csv, index=False)
                notify_step(res_b1)
            except Exception as e:
                print(f"Warning: Failed to evaluate {app_b1_name}: {e}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
    
    # 3. Approach B2 (Regressors benchmark with Standard LSTM Feature Extractor)
    reg_predictions = {}
    y_true_all = None
    for r_name in reg_models:
        app_b2_name = f"Approach B2 (Regressor {r_name})"
        if not should_skip(app_b2_name):
            print(f"--- Task 2 Running: {app_b2_name} ---")
            try:
                res_b2, y_t, y_p, _ = run_task2_approach_b2_regression(
                    train_df, test_df, r_name,
                    use_gpu=use_gpu, X_train_vec=X_train_lstm_combined, X_test_vec=X_test_lstm_combined,
                    lstm_tagger=lstm_tagger,
                    lstm_token_cache=lstm_token_cache,
                    token_cache=token_cache
                )
                results.append(res_b2)
                reg_predictions[r_name] = y_p
                y_true_all = y_t
                pd.DataFrame(results).to_csv(task2_csv, index=False)
                notify_step(res_b2)
            except Exception as e:
                print(f"Warning: Failed to evaluate {app_b2_name}: {e}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    # 4. Approach B3 (Binned Classifiers with BiLSTM-CRF Feature Extractor)
    for clf_name in clf_models:
        app_b3_name = f"Approach B3 (Binned {clf_name} + BiLSTM-CRF)"
        if not should_skip(app_b3_name):
            print(f"--- Task 2 Running: {app_b3_name} ---")
            try:
                res_b3, _ = run_task2_approach_b3_binned_bilstm_crf(
                    train_df, test_df, clf_name,
                    use_gpu=use_gpu, X_train_vec=X_train_bilstm_combined, X_test_vec=X_test_bilstm_combined,
                    bilstm_tagger=bilstm_tagger,
                    bilstm_token_cache=bilstm_token_cache,
                    token_cache=token_cache
                )
                results.append(res_b3)
                pd.DataFrame(results).to_csv(task2_csv, index=False)
                notify_step(res_b3)
            except Exception as e:
                print(f"Warning: Failed to evaluate {app_b3_name}: {e}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    # 5. Approach B4 (Regressors benchmark with BiLSTM-CRF Feature Extractor)
    for r_name in reg_models:
        app_b4_name = f"Approach B4 (Regressor {r_name} + BiLSTM-CRF)"
        if not should_skip(app_b4_name):
            print(f"--- Task 2 Running: {app_b4_name} ---")
            try:
                res_b4, y_t4, y_p4, _ = run_task2_approach_b4_regression_bilstm_crf(
                    train_df, test_df, r_name,
                    use_gpu=use_gpu, X_train_vec=X_train_bilstm_combined, X_test_vec=X_test_bilstm_combined,
                    bilstm_tagger=bilstm_tagger,
                    bilstm_token_cache=bilstm_token_cache,
                    token_cache=token_cache
                )
                results.append(res_b4)
                pd.DataFrame(results).to_csv(task2_csv, index=False)
                notify_step(res_b4)
            except Exception as e:
                print(f"Warning: Failed to evaluate {app_b4_name}: {e}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
    # 6. Approach C (Hybrid System) - Evaluate Hybrid Matrix across all Regressors with best Sequence Tagger
    best_pred_entities = bilstm_tagger.predict_entities(test_df["generated_text"].tolist())
    for r_name in reg_models:
        app_c_name = f"Approach C (Hybrid Rules + BiLSTM-CRF + {r_name})"
        if not should_skip(app_c_name):
            print(f"--- Task 2 Running: {app_c_name} ---")
            try:
                res_c = run_task2_approach_c_hybrid(
                    train_df, test_df,
                    best_regressor_name=r_name,
                    best_token_model_name="BiLSTM-CRF",
                    use_gpu=use_gpu, X_train_vec=X_train_bilstm_combined, X_test_vec=X_test_bilstm_combined,
                    best_pred_entities=best_pred_entities,
                    token_cache=token_cache
                )
                res_c["approach"] = app_c_name
                results.append(res_c)
                pd.DataFrame(results).to_csv(task2_csv, index=False)
                notify_step(res_c)
            except Exception as e:
                print(f"Warning: Failed to evaluate {app_c_name}: {e}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
    
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
