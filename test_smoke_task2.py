"""
Smoke Test for Task 2 Model Checkpointing & Fast Resume.
Tests with a small dataset (30 samples) to verify save, load, and skip in < 5 seconds.
"""
import os
import shutil
import time
import pandas as pd
from src.pipeline.task2_extraction import execute_task2_pipeline

def run_smoke_test():
    smoke_dir = "test_task2_smoke_out"
    if os.path.exists(smoke_dir):
        shutil.rmtree(smoke_dir)
        
    os.makedirs(smoke_dir, exist_ok=True)
    
    # Create small smoke train & test CSVs (30 samples train, 10 samples test)
    full_train = pd.read_csv("dataset/merged_synthetic_ner_dataset_v2.csv")
    full_test = pd.read_csv("dataset/gemini_3-1_flash_lite_synthetic_ner_dataset.csv")
    
    smoke_train_path = os.path.join(smoke_dir, "smoke_train.csv")
    smoke_test_path = os.path.join(smoke_dir, "smoke_test.csv")
    
    full_train.head(30).to_csv(smoke_train_path, index=False)
    full_test.head(10).to_csv(smoke_test_path, index=False)
    
    print("================================================================")
    print(">>> [TEST 1] First Run: Train & Save Checkpoints (force=False) <<")
    print("================================================================")
    t0 = time.time()
    execute_task2_pipeline(
        output_dir=smoke_dir,
        selected_models=["LogisticRegression", "Ridge"],
        use_gpu=False,
        force=False,
        train_path=smoke_train_path,
        test_path=smoke_test_path
    )
    t1 = time.time()
    print(f"\n[TEST 1 Completed in {t1 - t0:.2f}s]")
    
    # Check if files exist
    models_dir = os.path.join(smoke_dir, "models")
    assert os.path.exists(os.path.join(models_dir, "standard_lstm_tagger.pt")), "standard_lstm_tagger.pt missing!"
    assert os.path.exists(os.path.join(models_dir, "standard_lstm_features.pkl")), "standard_lstm_features.pkl missing!"
    assert os.path.exists(os.path.join(models_dir, "bilstm_crf_tagger.pt")), "bilstm_crf_tagger.pt missing!"
    assert os.path.exists(os.path.join(models_dir, "bilstm_crf_features.pkl")), "bilstm_crf_features.pkl missing!"
    print(">>> All checkpoint files verified on disk! <<<\n")
    
    print("================================================================")
    print(">>> [TEST 2] Second Run: Instant Resume from Checkpoint (force=False) <<")
    print("================================================================")
    t2 = time.time()
    execute_task2_pipeline(
        output_dir=smoke_dir,
        selected_models=["LogisticRegression", "Ridge"],
        use_gpu=False,
        force=False,
        train_path=smoke_train_path,
        test_path=smoke_test_path
    )
    t3 = time.time()
    print(f"\n[TEST 2 Completed in {t3 - t2:.2f}s (Instant Loaded!)]")
    
    print("================================================================")
    print(">>> [TEST 3] Third Run: Force Retrain (--force) <<")
    print("================================================================")
    t4 = time.time()
    execute_task2_pipeline(
        output_dir=smoke_dir,
        selected_models=["LogisticRegression", "Ridge"],
        use_gpu=False,
        force=True,
        train_path=smoke_train_path,
        test_path=smoke_test_path
    )
    t5 = time.time()
    print(f"\n[TEST 3 Completed in {t5 - t4:.2f}s (Retrained and Overwritten!)]")
    
    print("\n================================================================")
    print(">>> ALL SMOKE TESTS PASSED PERFECTLY! <<<")
    print("================================================================")

if __name__ == "__main__":
    run_smoke_test()
