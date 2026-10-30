"""Price the E1/E1b/E5 sweeps from measured pilot throughput (run after e0_pilot.sh adamw and muon).

    gw1b python team3/experiments/estimate_cost.py $GW1B_SCRATCH/teams/team3/runs/team3-e0-pilot-adamw-50m \\
        --muon $GW1B_SCRATCH/teams/team3/runs/team3-e0-pilot-muon-50m

Uses the median tokens/s of the pilot after its first log lines (compilation), scales to 100M and 200M by FLOPs
per token, and prints GPU-hours and wall-clock per run for the planned V100 sweeps (E1, E1b, E5).
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gw1b.budget import flops_per_token  # noqa: E402
from gw1b.config import load_config  # noqa: E402


def pilot_throughput(run_dir: str) -> tuple[float, int, str]:
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "metrics.jsonl"))]
    tps = [r["perf/tokens_per_s"] for r in rows if "perf/tokens_per_s" in r][2:]   # skip compile/warm-up
    if len(tps) < 3:
        raise SystemExit(f"{run_dir}: not enough logged steps to measure throughput")
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    return statistics.median(tps), summary["n_devices"], summary["gpu_kind"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adamw_pilot", help="run dir of the 50M AdamW pilot")
    ap.add_argument("--muon", help="run dir of the 50M Muon pilot (default: assume AdamW throughput)")
    a = ap.parse_args()

    tps_a, n_dev, kind = pilot_throughput(a.adamw_pilot)
    tps_m = pilot_throughput(a.muon)[0] if a.muon else tps_a
    cfg50, cfg100 = load_config("configs/team3/e1_50m.yaml"), load_config("configs/team3/e1b_100m.yaml")
    cfg200 = load_config("configs/team3/ladder_200m.yaml")
    fpt = lambda c: flops_per_token(c.model.n_params, c.model.n_layers, c.data.seq_len, c.model.d_model)
    scale_100 = fpt(cfg50) / fpt(cfg100)
    scale_200 = fpt(cfg50) / fpt(cfg200)
    tok50 = cfg50.resolved_total_steps() * cfg50.tokens_per_step
    tok100 = cfg100.resolved_total_steps() * cfg100.tokens_per_step
    tok200 = cfg200.resolved_total_steps() * cfg200.tokens_per_step

    print(f"pilot: {n_dev} x {kind}; AdamW {tps_a/1e3:,.1f}k tok/s, Muon {tps_m/1e3:,.1f}k tok/s "
          f"(Muon/AdamW step time {tps_a/tps_m:.2f}x)")
    plan = [  # (label, runs, tokens per run, tokens/s)
        ("E1 AdamW, 50M (11 tuning + 2 seeds)", 13, tok50, tps_a),
        ("E1 Muon, 50M (11 tuning + 2 seeds)", 13, tok50, tps_m),
        ("E1b AdamW, 100M (3 sweep + 1 WSD)", 4, tok100, tps_a * scale_100),
        ("E1b Muon, 100M (3 sweep + 1 WSD)", 4, tok100, tps_m * scale_100),
        ("E5 AdamW, 200M (3 sweep)", 3, tok200, tps_a * scale_200),
        ("E5 Muon, 200M (3 sweep)", 3, tok200, tps_m * scale_200),
    ]
    total = 0.0
    for label, n, tokens, tps in plan:
        hours = tokens / tps / 3600
        gpu_h = n * hours * n_dev
        total += gpu_h
        print(f"{label:<40} {n:>2} runs x {hours:5.1f} h wall on {n_dev} GPUs = {gpu_h:8,.0f} GPU-hours")
    print(f"{'total':<40} {total:31,.0f} GPU-hours")


if __name__ == "__main__":
    main()
