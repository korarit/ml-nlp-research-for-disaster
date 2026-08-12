"""
Data Loader Utility for Disaster Social Media Text Datasets.
Handles loading Gemini (5-Fold CV) and Luna (Held-out Benchmark) datasets.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any

DEFAULT_GEMINI_PATH = "dataset/gemini_3-1_flash_lite_synthetic_ner_dataset.csv"
DEFAULT_LUNA_PATH = "dataset/gpt_5_6_luna_paired_synthetic_ner_dataset.csv"

COUNT_COLUMNS = [
    "gt_dead", "gt_critical", "gt_urgent", "gt_safe",
    "gt_child", "gt_bedridden", "gt_item_firstaid",
    "gt_item_food", "gt_item_energy"
]


def load_dataset(filepath: str) -> pd.DataFrame:
    """Loads CSV dataset with UTF-8 encoding and fills standard NaNs."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    df = pd.read_csv(filepath, encoding="utf-8")
    
    # Fill count NAs with 0
    for col in COUNT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
            
    # Binary help request
    if "gt_is_help_request" in df.columns:
        df["gt_is_help_request_num"] = df["gt_is_help_request"].astype(int)
    elif "gt_classification_category" in df.columns:
        df["gt_is_help_request_num"] = (df["gt_classification_category"] == "help_request").astype(int)
        
    return df


def load_all_datasets(
    gemini_path: str = DEFAULT_GEMINI_PATH,
    luna_path: str = DEFAULT_LUNA_PATH
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads both Gemini CV dataset and Luna held-out test dataset."""
    gemini_df = load_dataset(gemini_path)
    luna_df = load_dataset(luna_path)
    return gemini_df, luna_df


def bin_count_target(y_counts: np.ndarray) -> np.ndarray:
    """Bins continuous counts into categorical classes: '0', '1', '2', '3+'."""
    binned = []
    for val in y_counts:
        v = int(val)
        if v == 0:
            binned.append("0")
        elif v == 1:
            binned.append("1")
        elif v == 2:
            binned.append("2")
        else:
            binned.append("3+")
    return np.array(binned)


def parse_victims_json(raw_json: Any) -> List[Dict[str, Any]]:
    """Parses gt_victims_json column safely into python list of dicts."""
    if pd.isna(raw_json) or not raw_json:
        return []
    if isinstance(raw_json, list):
        return raw_json
    try:
        data = json.loads(str(raw_json))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def extract_triage_data_by_age(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts individual victim triage samples from dataset, routing into:
    1. Pediatric Triage (Age <= 12)
    2. Adult Triage (Age > 12 / General)
    Returns DataFrames with columns: ['text', 'triage_color', 'age', 'symptoms_literal', 'victim_info']
    """
    pedia_records = []
    adult_records = []
    
    for idx, row in df.iterrows():
        text = row.get("generated_text", "")
        victims = parse_victims_json(row.get("gt_victims_json", ""))
        
        for vic in victims:
            color = vic.get("triage_color", "GREEN")
            if not color or color not in ["RED", "YELLOW", "GREEN"]:
                color = "GREEN"
                
            age = vic.get("age", None)
            symptoms = vic.get("symptoms_literal", "") or text
            
            record = {
                "text": text,
                "symptoms_literal": symptoms,
                "triage_color": color,
                "age": age,
                "is_child": vic.get("age_group") == "child" or (age is not None and age <= 12),
                "victim_info": vic
            }
            
            if record["is_child"]:
                pedia_records.append(record)
            else:
                adult_records.append(record)
                
    pedia_df = pd.DataFrame(pedia_records) if pedia_records else pd.DataFrame(columns=["text", "symptoms_literal", "triage_color", "age", "is_child", "victim_info"])
    adult_df = pd.DataFrame(adult_records) if adult_records else pd.DataFrame(columns=["text", "symptoms_literal", "triage_color", "age", "is_child", "victim_info"])
    
    return pedia_df, adult_df
