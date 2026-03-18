#!/usr/bin/env python3
"""
Repeatedly execute main.py (and whatever it configures) N times with specified seeds,
collect train_test outputs, summarize them, and save everything to JSON.
"""
import argparse
import json
import runpy
import time
from pathlib import Path

import numpy as np
import torch
import train_test_GCN as train_test_module


def summarize(all_runs):
    metrics = list(all_runs[0].keys())
    summary = {}
    for name in metrics:
        values = np.array([run[name]["value"] for run in all_runs], dtype=float)
        epochs = [run[name]["epoch"] for run in all_runs]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        best_idx = int(np.argmax(values))
        summary[name] = {
            "values": values.tolist(),
            "epochs": epochs,
            "mean": mean,
            "std": std,
            "best_run_index": best_idx,
            "best_value": float(values[best_idx]),
            "best_epoch": epochs[best_idx],
        }
    return summary


def run_main_once(main_path, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    capture = {}

    original_train_test = train_test_module.train_test

    def wrapped_train_test(*args, **kwargs):
        result = original_train_test(*args, **kwargs)
        capture["metrics"] = result
        return result

    train_test_module.train_test = wrapped_train_test
    try:
        runpy.run_path(str(main_path), run_name="__main__")
    finally:
        train_test_module.train_test = original_train_test
    if "metrics" not in capture:
        raise RuntimeError("main.py 未调用 train_test，无法捕获结果")
    return capture["metrics"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", default="main.py", help="要执行的 main.py 路径")
    parser.add_argument("--seeds", type=int, nargs='+', help="要使用的具体随机种子列表，例如: --seeds 42 95 127")
    parser.add_argument("--num-seeds", type=int, default=5, help="如果没有提供--seeds，则生成的随机种子个数")
    parser.add_argument("--base-seed", type=int, default=33, help="首个随机种子")
    parser.add_argument("--out", default="run_seed_results.json", help="结果 JSON 输出路径")
    args = parser.parse_args()

    main_path = Path(args.main).resolve()
    if not main_path.exists():
        raise FileNotFoundError(main_path)
        
    if args.seeds:
        seed_list = args.seeds
    else:
        seed_list = [args.base_seed + i for i in range(args.num_seeds)]

    per_run = []
    for idx, seed in enumerate(seed_list):
        print(f"\n=== Run {idx + 1} / {len(seed_list)} (seed={seed}) ===")
        start = time.time()
        metrics = run_main_once(main_path, seed)
        elapsed = time.time() - start
        print(f"完成，用时 {elapsed:.1f}s")
        per_run.append(metrics)

    summary = summarize(per_run)

    print("\n=== 汇总 ===")
    for name, info in summary.items():
        print(f"\n指标: {name}")
        print(f" 值: {info['values']}")
        print(f" Epochs: {info['epochs']}")
        print(f" Mean: {info['mean']:.4f}")
        print(f" Std: {info['std']:.6f}")
        print(f" 最佳: 第 {info['best_run_index'] + 1} 次, value={info['best_value']:.4f}, epoch={info['best_epoch']}")
    with open(args.out, "w") as fh:
        json.dump({"per_run": per_run, "summary": summary}, fh, indent=2)
    print(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()
