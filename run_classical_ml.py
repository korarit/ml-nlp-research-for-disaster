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
    parser.add_argument("--run_id", type=str, default="run_02_autotune_hyperparams",
                        help="Experiment run ID directory (e.g. run_01_baseline_default, run_02_autotune_hyperparams, or custom name)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom base directory to save results. Default: results/classical_ml/tuning_history/<run_id>")
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
    
    if args.output_dir:
        if os.path.basename(os.path.normpath(args.output_dir)) == args.run_id:
            base_output_dir = args.output_dir
        else:
            base_output_dir = os.path.join(args.output_dir, args.run_id)
    else:
        base_output_dir = os.path.join("results", "classical_ml", "tuning_history", args.run_id)
        
    os.makedirs(base_output_dir, exist_ok=True)
    
    print("==========================================================================")
    print(f" EXPERIMENT RUN ID: {args.run_id}")
    print(f" Task: {args.task} | Model: {args.model} | GPU Enabled: {args.use_gpu}")
    print(f" Optuna Trials: {args.n_trials} | CV Folds: {args.cv_folds} | Latency Runs: {args.latency_runs}")
    print("==========================================================================")
    
    notifier.notify_experiment_start(
        run_id=args.run_id,
        task=args.task,
        model=args.model,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds,
        use_gpu=args.use_gpu
    )
    
    selected_models = None if args.model.lower() == "all" else [args.model]
    
    try:
        # Task 1: Disaster Classification
        if args.task in ["1", "all"]:
            print("\n[Executing Task 1: Disaster Tweet Classification]")
            task1_df = execute_task1_pipeline(
                output_dir=base_output_dir,
                selected_models=selected_models,
                use_gpu=args.use_gpu,
                n_trials=args.n_trials,
                n_splits=args.cv_folds,
                latency_runs=args.latency_runs,
                force=args.force,
                notifier=notifier
            )
            print("Task 1 Completed successfully.")
            
        # Task 2: Extraction & Count Regression
        if args.task in ["2", "all"]:
            print("\n[Executing Task 2: NER & Entity/Count Extraction]")
            task2_df = execute_task2_pipeline(
                output_dir=base_output_dir,
                use_gpu=args.use_gpu,
                force=args.force,
                notifier=notifier
            )
            print("Task 2 Completed successfully.")
            
        # Task 3: Pediatric & Adult Triage
        if args.task in ["3.1", "3.2", "3.3", "3", "all"]:
            print("\n[Executing Task 3: Clinical Triage Classification (Pediatric & Adult)]")
            task3_df = execute_task3_pipeline(
                output_dir=base_output_dir,
                use_gpu=args.use_gpu,
                selected_classifiers=selected_models,
                force=args.force,
                notifier=notifier
            )
            print("Task 3 Completed successfully.")
            
        # Task 4: End-to-End Pipeline
        if args.task in ["4", "all"]:
            print("\n[Executing Task 4: End-to-End Integrated Pipeline Benchmark]")
            task4_res = execute_task4_e2e_pipeline(
                output_dir=base_output_dir,
                use_gpu=args.use_gpu,
                latency_runs=args.latency_runs,
                force=args.force,
                notifier=notifier
            )
            print("Task 4 Completed successfully. Results:")
            print(json.dumps(task4_res, indent=2))
            
        print("\n==========================================================================")
        print(f" ALL EXPERIMENTS COMPLETED SUCCESSFULLY. Results saved in:")
        print(f" file:///{os.path.abspath(base_output_dir)}")
        print("==========================================================================")
        
        notifier.notify_experiment_complete(args.run_id, os.path.abspath(base_output_dir))
        
    except Exception as e:
        notifier.notify_error(f"Task {args.task}", str(e))
        raise e


if __name__ == "__main__":
    main()

