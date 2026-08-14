"""
Master CLI Script Architecture implementing Section 8 of plan_exp_make_ml.md.
Supports modular execution, auto-checkpointing, GPU acceleration, Optuna tuning,
statistical significance tests, latency benchmarking, and graph generation.

Usage Examples:
    python run_classical_ml.py --task 1 --model XGBClassifier --use_gpu true --run_id run_02_autotune_hyperparams
    python run_classical_ml.py --task all --model all --use_gpu true --run_id run_03_final_luna_benchmark
    python run_classical_ml.py --task 4 --use_gpu true --run_id run_03_final_luna_benchmark
"""

import os
import sys
import argparse
import pandas as pd
import json

from src.pipeline.task1_classification import execute_task1_pipeline
from src.pipeline.task2_extraction import execute_task2_pipeline
from src.pipeline.task3_triage import execute_task3_pipeline
from src.pipeline.task4_e2e import execute_task4_e2e_pipeline
from src.utils.discord_notifier import DiscordNotifier


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Disaster NLP Classical ML & BiLSTM-CRF Benchmark CLI")
    
    parser.add_argument("--task", type=str, default="all", choices=["1", "2", "3", "3.1", "3.2", "3.3", "4", "all"],
                        help="Select Task to execute (1, 2, 3, 3.1, 3.2, 3.3, 4, or all)")
    parser.add_argument("--model", type=str, default="all",
                        help="Model name (e.g. XGBClassifier, LinearSVC, BiLSTM_CRF, or all)")
    parser.add_argument("--use_gpu", type=str2bool, default=True,
                        help="Enable GPU acceleration (true/false)")
    parser.add_argument("--train_path", type=str, default="dataset/merged_synthetic_ner_dataset_v2.csv",
                        help="Path to training dataset CSV")
    parser.add_argument("--test_path_1", type=str, default="dataset/gemini_3-1_flash_lite_synthetic_ner_dataset.csv",
                        help="Path to test dataset round 1 CSV (or 'none' to skip)")
    parser.add_argument("--test_path_2", type=str, default="dataset/gpt_5_6_luna_paired_synthetic_ner_dataset.csv",
                        help="Path to test dataset round 2 CSV (or 'none' to skip)")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Experiment run ID directory (default: automatically set to test_1 / test_2)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom base directory to save results.")
    parser.add_argument("--n_trials", type=int, default=15,
                        help="Number of Optuna tuning trials per model")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of Cross-Validation folds")
    parser.add_argument("--latency_runs", type=int, default=500,
                        help="Number of Latency measurement runs (N=1000 protocol)")
    parser.add_argument("--force", action="store_true",
                        help="Force rerun and overwrite existing checkpoints")
    parser.add_argument("--generate_graphs", action="store_true",
                        help="Generate standard visualization graphs")
    parser.add_argument("--run_stat_tests", action="store_true",
                        help="Run statistical significance tests (McNemar + Holm & Wilcoxon)")
    parser.add_argument("--discord_webhook", type=str, default=None,
                        help="Discord Webhook URL for real-time progress updates (or set DISCORD_WEBHOOK_URL in .env)")
                        
    return parser.parse_args()


def main():
    args = parse_arguments()
    notifier = DiscordNotifier(webhook_url=args.discord_webhook)
    
    # Collect valid test rounds (skip any set to None or "none")
    test_rounds = []
    if args.test_path_1 and str(args.test_path_1).lower() not in ("none", "null", ""):
        test_rounds.append(("test_1", args.test_path_1))
    if args.test_path_2 and str(args.test_path_2).lower() not in ("none", "null", ""):
        test_rounds.append(("test_2", args.test_path_2))
        
    if not test_rounds:
        print("[WARNING] Both test_path_1 and test_path_2 are set to None. No evaluation test sets specified.")
        return

    selected_models = None if args.model.lower() == "all" else [args.model]

    for round_name, test_path in test_rounds:
        run_id = args.run_id or round_name
        if args.output_dir:
            base_output_dir = os.path.join(args.output_dir, round_name)
        else:
            base_output_dir = round_name

        os.makedirs(base_output_dir, exist_ok=True)
        os.makedirs(os.path.join(base_output_dir, "graphs"), exist_ok=True)
        
        print("\n==========================================================================")
        print(f" EXPERIMENT EVALUATION ROUND: [{round_name.upper()}] (Folder: {base_output_dir})")
        print(f" Train Path: {args.train_path}")
        print(f" Test Path:  {test_path}")
        print(f" Task: {args.task} | Model: {args.model} | GPU Enabled: {args.use_gpu}")
        print(f" Optuna Trials: {args.n_trials} | CV Folds: {args.cv_folds} | Latency Runs: {args.latency_runs}")
        print("==========================================================================")
        
        notifier.notify_experiment_start(
            run_id=run_id,
            task=args.task,
            model=args.model,
            n_trials=args.n_trials,
            cv_folds=args.cv_folds,
            use_gpu=args.use_gpu
        )
        
        try:
            # Task 1: Disaster Classification
            if args.task in ["1", "all"]:
                print(f"\n[Executing Task 1 for {round_name}: Disaster Tweet Classification]")
                task1_df = execute_task1_pipeline(
                    output_dir=base_output_dir,
                    selected_models=selected_models,
                    use_gpu=args.use_gpu,
                    n_trials=args.n_trials,
                    n_splits=args.cv_folds,
                    latency_runs=args.latency_runs,
                    force=args.force,
                    notifier=notifier,
                    train_path=args.train_path,
                    test_path=test_path
                )
                print(f"Task 1 ({round_name}) Completed successfully.")
                
            # Task 2: Extraction & Count Regression
            if args.task in ["2", "all"]:
                print(f"\n[Executing Task 2 for {round_name}: NER & Entity/Count Extraction]")
                task2_df = execute_task2_pipeline(
                    output_dir=base_output_dir,
                    selected_models=selected_models,
                    use_gpu=args.use_gpu,
                    force=args.force,
                    notifier=notifier,
                    train_path=args.train_path,
                    test_path=test_path
                )
                print(f"Task 2 ({round_name}) Completed successfully.")
                
            # Task 3: Pediatric & Adult Triage
            if args.task in ["3.1", "3.2", "3.3", "3", "all"]:
                print(f"\n[Executing Task 3 for {round_name}: Clinical Triage Classification (Pediatric & Adult)]")
                task3_df = execute_task3_pipeline(
                    output_dir=base_output_dir,
                    use_gpu=args.use_gpu,
                    selected_classifiers=selected_models,
                    force=args.force,
                    notifier=notifier,
                    train_path=args.train_path,
                    test_path=test_path
                )
                print(f"Task 3 ({round_name}) Completed successfully.")
                
            # Task 4: End-to-End Pipeline
            if args.task in ["4", "all"]:
                print(f"\n[Executing Task 4 for {round_name}: End-to-End Integrated Pipeline Benchmark]")
                task4_res = execute_task4_e2e_pipeline(
                    output_dir=base_output_dir,
                    use_gpu=args.use_gpu,
                    latency_runs=args.latency_runs,
                    force=args.force,
                    notifier=notifier,
                    train_path=args.train_path,
                    test_path=test_path
                )
                print(f"Task 4 ({round_name}) Completed successfully. Results:")
                print(json.dumps(task4_res, indent=2))
                
            print("\n==========================================================================")
            print(f" EVALUATION ROUND [{round_name.upper()}] COMPLETED. Results saved in:")
            print(f" file:///{os.path.abspath(base_output_dir)}")
            print("==========================================================================")
            
            notifier.notify_experiment_complete(run_id, os.path.abspath(base_output_dir))
            
        except Exception as e:
            notifier.notify_error(f"Task {args.task} [{round_name}]", str(e))
            raise e


if __name__ == "__main__":
    main()

