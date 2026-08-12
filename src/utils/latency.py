"""
Inference Latency Measurement Suite implementing Section 4.2 Protocol.
Measures empirical distribution (Mean, p50, p95, p99, QPS) on single-sample streaming input.
Supports GPU (CUDA sync) and CPU (Single-thread forced) environments.
"""

import time
import numpy as np
import torch
from typing import Dict, Any, Callable


def measure_inference_latency(
    predict_fn: Callable[[str], Any],
    sample_text: str = "ช่วยด้วยค่ะ มีผู้ป่วยผู้ใหญ่ 2 คน ติดค้างน้ำท่วมหนัก อาหารหมด โทร 0812345678",
    n_runs: int = 1000,
    warmup_runs: int = 100,
    use_gpu: bool = False
) -> Dict[str, float]:
    """
    Executes Section 4.2 Latency Protocol:
    1. Warm-up (100 iterations)
    2. Single-sample streaming input (batch size = 1)
    3. CUDA Synchronization (if GPU enabled)
    4. CPU Single-thread enforcement (if CPU mode)
    5. Computes Empirical Latency Distribution (Mean, p50, p95, p99, QPS)
    """
    orig_threads = torch.get_num_threads()
    try:
        if not use_gpu:
            torch.set_num_threads(1)
            
        # 1. Warm-up Phase
        for _ in range(warmup_runs):
            _ = predict_fn(sample_text)
            if use_gpu and torch.cuda.is_available():
                torch.cuda.synchronize()
                
        # 2. Latency Measurement Phase
        latency_ms_list = []
        
        for _ in range(n_runs):
            if use_gpu and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            _ = predict_fn(sample_text)
            
            if use_gpu and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            elapsed_ms = (t1 - t0) * 1000.0
            latency_ms_list.append(elapsed_ms)
    finally:
        if not use_gpu:
            torch.set_num_threads(orig_threads)
            
    latencies = np.array(latency_ms_list)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))
    qps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0
    
    return {
        "mean_latency_ms": mean_lat,
        "std_latency_ms": std_lat,
        "p50_latency_ms": p50_lat,
        "p95_latency_ms": p95_lat,
        "p99_latency_ms": p99_lat,
        "qps": qps
    }
