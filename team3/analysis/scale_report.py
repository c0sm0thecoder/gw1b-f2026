"""AdamW vs Muon across model sizes: tables and plots from the Team 3 run directories.

    gw1b python team3/analysis/scale_report.py $GW1B_SCRATCH/teams/team3/runs --out team3_report
    gw1b python team3/analysis/scale_report.py <runs dir> --prefix team3-e1- --out e1_report   # one experiment

Reads every run directory (config.yaml + metrics.jsonl, summary.json once finished) and groups runs by model size,
precision and schedule. Per group it picks each optimizer's best setting (lowest final validation loss at seed 0,
finished runs only) and averages its seeds. Writes to --out:

    runs.csv            one row per run: optimizer, lr, wd, seed, schedule, size, final val loss, step time
    best.csv            per group and optimizer: best lr / wd, mean and spread over seeds, number of seeds
    multipliers.csv     per group: share of AdamW's tokens Muon needs to reach AdamW's final loss, and the same in
                        wall-clock (tokens share x step-time ratio)
    curves_<group>.png  validation loss vs tokens of each optimizer's best setting, every seed
    lr_<group>.png      final validation loss vs peak learning rate, per optimizer and weight decay
    scale.png           best loss and Muon's token share vs model size (the scale trend)

Muon's token share reads Muon's validation curve partway through its own schedule, before its learning rate has
decayed, so it is conservative: a Muon run scheduled for that shorter length would get there sooner.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gw1b.config import load_config  # noqa: E402

OPTIMIZERS = ("adamw", "muon")
COLORS = {"adamw": "tab:blue", "muon": "tab:red"}


def load_run(run_dir: str) -> dict | None:
    """One run: settings from config.yaml, curves from metrics.jsonl. None if the run has no validation point yet."""
    if not (os.path.exists(os.path.join(run_dir, "config.yaml")) and os.path.exists(os.path.join(run_dir, "metrics.jsonl"))):
        return None
    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "metrics.jsonl")) if line.strip()]
    val = {}  # step -> (tokens, loss); a resumed run may log a step twice, keep the last
    for r in rows:
        if "val/loss" in r:
            val[r["step"]] = (r["tokens_seen"], r["val/loss"])
    if not val:
        return None
    step_times = [r["perf/step_time_s"] for r in rows if "perf/step_time_s" in r][2:]  # skip compilation
    summary_path = os.path.join(run_dir, "summary.json")
    finished = os.path.exists(summary_path) and json.load(open(summary_path)).get("finished", False)
    curve = np.array([val[s] for s in sorted(val)], dtype=float)
    return {
        "run": os.path.basename(run_dir.rstrip("/")),
        "optimizer": cfg.optim.name, "lr": cfg.optim.lr, "wd": cfg.optim.weight_decay, "seed": cfg.run.seed,
        "schedule": cfg.optim.schedule, "dtype": cfg.model.dtype, "params_m": round(cfg.model.n_params / 1e6),
        "tokens_planned": cfg.resolved_total_steps() * cfg.tokens_per_step,
        "finished": finished, "curve": curve, "final_val": float(curve[-1, 1]),
        "step_time_s": statistics.median(step_times) if step_times else float("nan"),
    }


def group_key(r: dict) -> str:
    return f"{r['params_m']}M-{r['dtype']}-{r['schedule']}-{r['tokens_planned'] / 1e9:g}Btok"


def tokens_to_reach(curve: np.ndarray, target: float) -> float | None:
    """First token count at which a validation curve reaches `target`, interpolated between evaluations."""
    below = np.nonzero(curve[:, 1] <= target)[0]
    if not len(below):
        return None
    i = below[0]
    if i == 0:
        return float(curve[0, 0])
    (t0, l0), (t1, l1) = curve[i - 1], curve[i]
    return float(t0 + (l0 - target) / (l0 - l1) * (t1 - t0))


def best_settings(runs: list[dict]) -> dict[str, dict]:
    """Per optimizer: the (lr, wd) with the lowest finished seed-0 loss, and all finished seeds of it."""
    out = {}
    for opt in OPTIMIZERS:
        seed0 = [r for r in runs if r["optimizer"] == opt and r["seed"] == 0 and r["finished"]]
        if not seed0:
            continue
        b = min(seed0, key=lambda r: r["final_val"])
        seeds = sorted((r for r in runs if r["optimizer"] == opt and r["finished"]
                        and r["lr"] == b["lr"] and r["wd"] == b["wd"]), key=lambda r: r["seed"])
        losses = [r["final_val"] for r in seeds]
        out[opt] = {"lr": b["lr"], "wd": b["wd"], "seeds": seeds, "mean": statistics.mean(losses),
                    "sd": statistics.stdev(losses) if len(losses) > 1 else float("nan"),
                    "step_time_s": statistics.median(r["step_time_s"] for r in seeds)}
    return out


def multiplier(best: dict[str, dict]) -> dict | None:
    """Muon's tokens (and time) to reach AdamW's mean final loss, as a share of AdamW's, per Muon seed."""
    if not all(o in best for o in OPTIMIZERS):
        return None
    a, m = best["adamw"], best["muon"]
    target, budget = a["mean"], a["seeds"][0]["curve"][-1, 0]
    shares = [t / budget for t in (tokens_to_reach(r["curve"], target) for r in m["seeds"]) if t is not None]
    step_ratio = m["step_time_s"] / a["step_time_s"]
    return {"target": target, "token_share": statistics.mean(shares) if shares else None,
            "token_share_min": min(shares) if shares else None, "token_share_max": max(shares) if shares else None,
            "muon_seeds_reaching": len(shares), "muon_seeds": len(m["seeds"]), "step_time_ratio": step_ratio,
            "time_share": statistics.mean(shares) * step_ratio if shares else None}


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot_group(key: str, runs: list[dict], best: dict[str, dict], out: str, plt) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for opt, b in best.items():
        for r in b["seeds"]:
            ax.plot(r["curve"][:, 0] / 1e9, r["curve"][:, 1], "-" if r["seed"] == 0 else "--", color=COLORS[opt],
                    marker="o", ms=3, label=f"{opt} lr={b['lr']:g} wd={b['wd']:g} seed {r['seed']}")
    ax.set(xlabel="training tokens (B)", ylabel="validation loss", title=f"{key}: best setting per optimizer")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"curves_{key}.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for opt in OPTIMIZERS:
        for wd in sorted({r["wd"] for r in runs if r["optimizer"] == opt}):
            pts = sorted((r["lr"], r["final_val"]) for r in runs
                         if r["optimizer"] == opt and r["wd"] == wd and r["seed"] == 0 and r["finished"])
            if pts:
                ax.plot(*zip(*pts), "o-", color=COLORS[opt], alpha=0.4 + 0.6 * (wd == best.get(opt, {}).get("wd")),
                        label=f"{opt} wd={wd:g}")
    ax.set(xscale="log", xlabel="peak learning rate", ylabel="final validation loss",
           title=f"{key}: learning-rate sensitivity (seed 0)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"lr_{key}.png"), dpi=150); plt.close(fig)


def plot_scale(rows: list[dict], mults: list[dict], out: str, plt) -> None:
    """The trend uses the cosine groups only (E2-style schedule checks are separate points). Raw losses are drawn for
    float32 sizes only; bf16 sizes (A100) enter only through the Muon/AdamW token share, which is a ratio."""
    rows = [r for r in rows if r["schedule"] == "cosine"]
    mults = [m for m in mults if m["schedule"] == "cosine"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for opt in OPTIMIZERS:
        pts = sorted((r["params_m"], r["mean"]) for r in rows if r["optimizer"] == opt and r["dtype"] == "float32")
        if pts:
            ax[0].plot(*zip(*pts), "o-", color=COLORS[opt], label=opt)
    ax[0].set(xscale="log", xlabel="parameters (M)", ylabel="best final validation loss (mean over seeds)",
              title="Best loss vs model size (float32 runs)")
    pts = sorted((m["params_m"], m["token_share"], m["token_share_min"], m["token_share_max"])
                 for m in mults if m["token_share"] is not None)
    if pts:
        x, y, lo, hi = map(np.array, zip(*pts))
        ax[1].errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", color="tab:red", capsize=4)
    ax[1].axhline(1.0, color="gray", ls=":")
    ax[1].set(xscale="log", xlabel="parameters (M)", ylabel="Muon tokens / AdamW tokens to reach AdamW's loss",
              title="Muon's token share vs model size (1 = no gain)")
    for a in ax:
        a.grid(alpha=.3)
    ax[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "scale.png"), dpi=150); plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="directories holding run directories")
    ap.add_argument("--prefix", default="team3-", help="only runs whose name starts with this (default team3-)")
    ap.add_argument("--out", default="team3_report", help="output directory")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)

    runs = [r for root in a.roots for d in sorted(glob.glob(os.path.join(root, a.prefix + "*")))
            if os.path.isdir(d) and (r := load_run(d)) is not None and r["optimizer"] in OPTIMIZERS]
    if not runs:
        raise SystemExit(f"no runs with validation points under {a.roots} matching {a.prefix}*")
    os.makedirs(a.out, exist_ok=True)
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(group_key(r), []).append(r)

    write_csv(os.path.join(a.out, "runs.csv"), [
        {k: v for k, v in r.items() if k != "curve"} | {"group": group_key(r)} for r in runs])
    best_rows, mult_rows = [], []
    print(f"{len(runs)} runs in {len(groups)} group(s)\n")
    for key in sorted(groups, key=lambda k: (groups[k][0]["params_m"], k)):
        best, g0 = best_settings(groups[key]), groups[key][0]
        for opt, b in best.items():
            best_rows.append({"group": key, "params_m": g0["params_m"], "dtype": g0["dtype"], "schedule": g0["schedule"],
                              "optimizer": opt, "lr": b["lr"], "wd": b["wd"], "mean": b["mean"], "sd": b["sd"],
                              "n_seeds": len(b["seeds"]), "step_time_s": b["step_time_s"]})
            print(f"{key:<28} {opt:<6} best lr {b['lr']:<8g} wd {b['wd']:<5g} "
                  f"val {b['mean']:.4f} (sd {b['sd']:.4f}, {len(b['seeds'])} seed(s))")
        m = multiplier(best)
        if m:
            mult_rows.append({"group": key, "params_m": g0["params_m"], "dtype": g0["dtype"], "schedule": g0["schedule"]} | m)
            share = f"{m['token_share']:.0%} of the tokens" if m["token_share"] else "never (within its run)"
            print(f"{'':<28} Muon reaches AdamW's {m['target']:.4f}: {share}; step time {m['step_time_ratio']:.2f}x"
                  + (f" -> {m['time_share']:.0%} of the time" if m["time_share"] else ""))
        print()
    write_csv(os.path.join(a.out, "best.csv"), best_rows)
    write_csv(os.path.join(a.out, "multipliers.csv"), mult_rows)

    if not a.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for key, group in groups.items():
            plot_group(key, group, best_settings(group), a.out, plt)
        plot_scale(best_rows, mult_rows, a.out, plt)
    print(f"tables and plots -> {a.out}/")


if __name__ == "__main__":
    main()
