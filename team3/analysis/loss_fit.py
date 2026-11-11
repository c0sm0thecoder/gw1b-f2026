"""Scaling fit: predict GW-1B's loss at 20 and 30 tokens/parameter, with uncertainty, from the ladder runs.

gw1b python team3/analysis/scale_report.py $GW1B_SCRATCH/teams/team3/runs --out team3_report --no-plots
gw1b python team3/analysis/scale_report.py $GW1B_SCRATCH/teams/team3/runs --prefix team3-e6- --out team3_report_e6 --no-plots
python team3/analysis/loss_fit.py team3_report/best.csv --token-best team3_report_e6/best.csv --out fit.csv

Input is best.csv from scale_report.py (each optimizer's best setting per model size, tokens/parameter, schedule). Needs
only numpy, so it also runs off the cluster. Per optimizer the prediction is separable:

size trend   f(N): L = E + a * N**-b fitted to the size ladder at --base-tpp (cosine runs). E is --irreducible, 0 by
             default, so the default is a plain power law, which is optimistic when extrapolating.
token axis   g(t): L(N, t) / L(N, base) at every size that has both, averaged (WSD runs: the E6 trunks and branches,
             so the schedule is the same at every t). The spread of the ratio across sizes is printed.
prediction   L(N_target, t) = f(N_target) * g(t)

Intervals (5th to 95th percentile) come from redrawing every loss from its seed noise (sd / sqrt(n_seeds); a single-seed
point uses the pooled sd of that optimizer's multi-seed points) and refitting. They cover measurement noise only, not
the choice of functional form or the distance of the extrapolation: the backtest line refits without the largest size
and shows how far off the extrapolation to that size was.

Quantities written to --out: loss (at each tokens/parameter), gain_from_prev (loss saved by the next token budget,
positive is better), muon_minus_adamw (negative means Muon is ahead), p_muon_better (share of redraws where it is).
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass

import numpy as np

OPTIMIZERS = ("adamw", "muon")
FIELDS = ["quantity", "optimizer", "tok_per_param", "value", "lo", "hi"]


@dataclass(frozen=True)
class Spec:
    target_params_m: float = 1150.0
    base_tpp: int = 10
    tpps: tuple[int, ...] = (20, 30)
    size_schedule: str = "cosine"
    token_schedule: str = "wsd"
    irreducible: float = 0.0


def parse_row(r: dict) -> dict:
    """One best.csv row as the numbers the fit uses; an empty sd (a single seed) is nan."""
    return {"params_m": float(r["params_m"]), "tpp": int(float(r["tok_per_param"])), "schedule": r["schedule"],
            "optimizer": r["optimizer"], "mean": float(r["mean"]),
            "sd": float("nan") if r["sd"] in ("", None) else float(r["sd"]), "n_seeds": int(r["n_seeds"])}


def load_best(path: str, dtype: str | None = "float32") -> list[dict]:
    """Rows of scale_report's best.csv; `dtype` keeps one precision so bf16 and float32 runs never mix."""
    with open(path, newline="") as f:
        return [parse_row(r) for r in csv.DictReader(f) if not dtype or r["dtype"] == dtype]


def use_token_runs(rows: list[dict], token_rows: list[dict], schedule: str) -> list[dict]:
    """`rows` with every `schedule` row replaced by the ones from `token_rows`.

    scale_report.py groups by schedule name only, so a WSD row of the full report can pool runs with other cooldown
    lengths (E2). A report of the E6 runs alone (--prefix team3-e6-) has one run per point, trunk-matched.
    """
    return [r for r in rows if r["schedule"] != schedule] + [r for r in token_rows if r["schedule"] == schedule]


def points(rows: list[dict], optimizer: str, schedule: str, tpp: int) -> dict[float, dict]:
    """Model size (M parameters) -> row, for one optimizer, schedule and tokens/parameter."""
    out: dict[float, dict] = {}
    for r in rows:
        if r["optimizer"] == optimizer and r["schedule"] == schedule and r["tpp"] == tpp:
            if r["params_m"] in out:
                raise ValueError(f"more than one {optimizer} {schedule} row at {r['params_m']:g}M, {tpp} tok/param; "
                                 "rerun scale_report.py on one experiment (--prefix) or pass --dtype")
            out[r["params_m"]] = r
    return out


def pooled_sd(rows: list[dict], optimizer: str) -> float:
    """Seed spread of an optimizer from its multi-seed points; 0 when there are none."""
    sds = [r["sd"] for r in rows if r["optimizer"] == optimizer and r["n_seeds"] > 1 and math.isfinite(r["sd"])]
    return math.sqrt(statistics.mean(s * s for s in sds)) if sds else 0.0


def std_error(row: dict, fallback_sd: float) -> float:
    sd = row["sd"] if row["n_seeds"] > 1 and math.isfinite(row["sd"]) else fallback_sd
    return sd / math.sqrt(row["n_seeds"])


def fit_power(sizes: list[float], losses: list[float], irreducible: float = 0.0) -> tuple[float, float]:
    """(a, b) of L = irreducible + a * N**-b, by least squares on log(L - irreducible) against log N."""
    if min(losses) <= irreducible:
        raise ValueError(f"--irreducible {irreducible:g} is not below every loss (lowest {min(losses):g})")
    slope, intercept = np.polyfit(np.log(sizes), np.log(np.asarray(losses) - irreducible), 1)
    return math.exp(intercept), -float(slope)


def power(a: float, b: float, size: float, irreducible: float = 0.0) -> float:
    return irreducible + a * size ** -b


def collect(rows: list[dict], optimizer: str, spec: Spec) -> tuple[dict[float, dict], dict[int, dict[float, float]]]:
    """The size ladder and, per tokens/parameter, the loss ratio to the base budget at each size that has both."""
    ladder = points(rows, optimizer, spec.size_schedule, spec.base_tpp)
    if len(ladder) < 2:
        raise ValueError(f"{optimizer}: need the {spec.size_schedule} ladder at {spec.base_tpp} tok/param at two or "
                         f"more sizes, found {len(ladder)}")
    base = points(rows, optimizer, spec.token_schedule, spec.base_tpp)
    ratios = {}
    for t in spec.tpps:
        at_t = points(rows, optimizer, spec.token_schedule, t)
        ratios[t] = {s: at_t[s]["mean"] / base[s]["mean"] for s in at_t if s in base}
        if not ratios[t]:
            raise ValueError(f"{optimizer}: no size has {spec.token_schedule} rows at both {spec.base_tpp} and {t} "
                             "tok/param")
    return ladder, ratios


def predict_optimizer(rows: list[dict], optimizer: str, spec: Spec) -> dict[int, float]:
    """Predicted loss at the target size for the base budget and each of spec.tpps."""
    ladder, ratios = collect(rows, optimizer, spec)
    sizes = sorted(ladder)
    a, b = fit_power(sizes, [ladder[s]["mean"] for s in sizes], spec.irreducible)
    base_loss = power(a, b, spec.target_params_m, spec.irreducible)
    return {spec.base_tpp: base_loss} | {t: base_loss * statistics.mean(r.values()) for t, r in ratios.items()}


def backtest(rows: list[dict], optimizer: str, spec: Spec) -> dict | None:
    """Refit the size trend without the largest size and predict it. None with fewer than three sizes."""
    ladder = points(rows, optimizer, spec.size_schedule, spec.base_tpp)
    if len(ladder) < 3:
        return None
    top = max(ladder)
    rest = sorted(s for s in ladder if s != top)
    a, b = fit_power(rest, [ladder[s]["mean"] for s in rest], spec.irreducible)
    predicted, observed = power(a, b, top, spec.irreducible), ladder[top]["mean"]
    return {"size_m": top, "predicted": predicted, "observed": observed, "error": predicted - observed}


def run_fit(rows: list[dict], spec: Spec, n_boot: int = 2000, seed: int = 0) -> dict:
    """Point predictions and bootstrap draws, {optimizer: {tokens/parameter: loss or array of n_boot losses}}."""
    for opt in OPTIMIZERS:
        if not any(r["optimizer"] == opt for r in rows):
            raise ValueError(f"no {opt} rows in the input")
    rng = np.random.default_rng(seed)
    fallback = {opt: pooled_sd(rows, opt) for opt in OPTIMIZERS}
    point = {opt: predict_optimizer(rows, opt, spec) for opt in OPTIMIZERS}
    draws: dict[str, dict[int, list[float]]] = {opt: {t: [] for t in point[opt]} for opt in OPTIMIZERS}
    for _ in range(n_boot):
        noisy = [r | {"mean": float(rng.normal(r["mean"], std_error(r, fallback[r["optimizer"]])))} for r in rows]
        for opt in OPTIMIZERS:
            for t, v in predict_optimizer(noisy, opt, spec).items():
                draws[opt][t].append(v)
    return {"point": point, "draws": {o: {t: np.array(v) for t, v in d.items()} for o, d in draws.items()},
            "fallback_sd": fallback}


def interval(draws: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(draws, [5, 95])
    return float(lo), float(hi)


def summarize(fit: dict, spec: Spec) -> list[dict]:
    """Table rows (FIELDS) of every quantity the script reports."""
    budgets = [spec.base_tpp, *sorted(spec.tpps)]
    point, draws = fit["point"], fit["draws"]
    rows = []

    def add(quantity, optimizer, tpp, value, d=None, with_interval=True):
        lo, hi = interval(d) if d is not None and with_interval else (None, None)
        rows.append({"quantity": quantity, "optimizer": optimizer, "tok_per_param": tpp, "value": value,
                     "lo": lo, "hi": hi})

    for opt in OPTIMIZERS:
        for t in budgets:
            add("loss", opt, t, point[opt][t], draws[opt][t])
        for prev, t in zip(budgets, budgets[1:]):
            add("gain_from_prev", opt, t, point[opt][prev] - point[opt][t], draws[opt][prev] - draws[opt][t])
    for t in budgets:
        gap = draws["muon"][t] - draws["adamw"][t]
        add("muon_minus_adamw", "", t, point["muon"][t] - point["adamw"][t], gap)
        add("p_muon_better", "", t, float(np.mean(gap < 0)))
    return rows


def write_table(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: "" if v is None else v for k, v in r.items()} for r in rows)


def report(rows: list[dict], fit: dict, spec: Spec, table: list[dict]) -> str:
    def span(r):
        return f"{r['value']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]"

    lines = [f"target {spec.target_params_m:g}M parameters; size trend: {spec.size_schedule} runs at {spec.base_tpp} "
             f"tok/param; token axis: {spec.token_schedule} runs; 90% intervals from seed noise only", ""]
    for opt in OPTIMIZERS:
        ladder, ratios = collect(rows, opt, spec)
        lines.append(f"{opt}: size ladder {', '.join(f'{s:g}M' for s in sorted(ladder))}")
        if len(ladder) < 3:
            lines.append("  only two sizes: the size trend has no spare point, treat the interval as a lower bound")
        for t, r in sorted(ratios.items()):
            vals = list(r.values())
            lines.append(f"  loss ratio {t}/{spec.base_tpp} tok/param over {len(vals)} size(s): "
                         f"{statistics.mean(vals):.4f} (range {min(vals):.4f} to {max(vals):.4f})")
        if not fit["fallback_sd"][opt]:
            lines.append("  no multi-seed points: seed noise taken as zero, intervals are degenerate")
        bt = backtest(rows, opt, spec)
        if bt:
            lines.append(f"  backtest: ladder without {bt['size_m']:g}M predicts {bt['predicted']:.4f} for it, "
                         f"observed {bt['observed']:.4f} (error {bt['error']:+.4f})")
    lines.append("")
    for q, title in (("loss", "predicted loss"), ("gain_from_prev", "loss saved by the next token budget"),
                     ("muon_minus_adamw", "Muon minus AdamW (negative: Muon ahead)")):
        lines.append(f"{title}:")
        for r in table:
            if r["quantity"] == q:
                lines.append(f"  {r['optimizer'] or 'gap':<7}{r['tok_per_param']:>3} tok/param  {span(r)}")
    p = ", ".join(f"{r['tok_per_param']} tok/param {r['value']:.2f}" for r in table if r["quantity"] == "p_muon_better")
    lines.append(f"share of redraws where Muon is ahead: {p}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("best_csv", help="best.csv written by scale_report.py")
    ap.add_argument("--target-params-m", type=float, default=1150.0, help="model size to predict, in M parameters")
    ap.add_argument("--base-tpp", type=int, default=10, help="tokens/parameter of the size ladder")
    ap.add_argument("--tpp", type=int, nargs="+", default=[20, 30], help="tokens/parameter to predict")
    ap.add_argument("--size-schedule", default="cosine", help="schedule of the size ladder")
    ap.add_argument("--token-schedule", default="wsd", help="schedule of the runs that give the token axis")
    ap.add_argument("--token-best", help="best.csv of scale_report.py --prefix team3-e6-: its --token-schedule rows "
                                         "replace those of best_csv, so the token axis uses only the E6 runs")
    ap.add_argument("--dtype", default="float32", help="precision of the runs to use ('' keeps all)")
    ap.add_argument("--irreducible", type=float, default=0.0, help="loss floor E in L = E + a * N**-b")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write the table here as csv")
    a = ap.parse_args(argv)

    spec = Spec(a.target_params_m, a.base_tpp, tuple(t for t in a.tpp if t != a.base_tpp), a.size_schedule,
                a.token_schedule, a.irreducible)
    rows = load_best(a.best_csv, a.dtype)
    if a.token_best:
        rows = use_token_runs(rows, load_best(a.token_best, a.dtype), a.token_schedule)
    fit = run_fit(rows, spec, a.n_boot, a.seed)
    table = summarize(fit, spec)
    print(report(rows, fit, spec, table))
    if a.out:
        write_table(a.out, table)
        print(f"\ntable -> {a.out}")


if __name__ == "__main__":
    main()
