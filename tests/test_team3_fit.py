"""Team 3 analysis: loss_fit.py on small best.csv tables. Run: python -m pytest tests -q"""
import csv
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "team3", "analysis"))

import loss_fit # noqa: E402

BEST_FIELDS = ["group", "params_m", "tok_per_param", "dtype", "schedule", "optimizer", "lr", "wd", "mean", "sd",
               "n_seeds", "step_time_s"]
SIZES = (50, 100, 200, 350)
A, B = 5.0, 0.15 # AdamW cosine ladder at 10 tok/param: loss = A * N**-B, N in M parameters


def row(size, tpp, optimizer, mean, sd=0.0, n=3, schedule="cosine", dtype="float32"):
    return {"group": f"{size}M-{tpp}tpp", "params_m": size, "tok_per_param": tpp, "dtype": dtype,
            "schedule": schedule, "optimizer": optimizer, "lr": 4.8e-3, "wd": 0.1, "mean": mean,
            "sd": "" if sd is None else sd, "n_seeds": n, "step_time_s": 1.0}


def write_best(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def exact_rows(sd=0.0, muon_factor=0.97):
    """Ladder from an exact power law, WSD token axis 1 / 0.95 / 0.92 at 10 / 20 / 30 tok/param (Muon below by 3%)."""
    rows = []
    for opt, k in (("adamw", 1.0), ("muon", muon_factor)):
        for s in SIZES:
            rows.append(row(s, 10, opt, k * A * s ** -B, sd))
        for s in (50, 100):
            base = k * A * s ** -B * 1.001 # WSD at 10 tok/param, a hair off the cosine run
            for tpp, g in ((10, 1.0), (20, 0.95), (30, 0.92)):
                rows.append(row(s, tpp, opt, base * g, sd, schedule="wsd"))
    return rows


def parsed(rows):
    return [loss_fit.parse_row(r) for r in rows]


def table(result):
    return {(r["quantity"], r["optimizer"], r["tok_per_param"]): r for r in result}


def test_power_law_is_recovered():
    a, b = loss_fit.fit_power([50, 100, 200, 350], [A * s ** -B for s in (50, 100, 200, 350)])
    assert (a, b) == pytest.approx((A, B))
    a, b = loss_fit.fit_power([50, 100, 200], [1.7 + 3 * s ** -0.2 for s in (50, 100, 200)], irreducible=1.7)
    assert (a, b) == pytest.approx((3.0, 0.2))


def test_prediction_is_size_trend_times_token_ratio():
    spec = loss_fit.Spec()
    pred = loss_fit.predict_optimizer(parsed(exact_rows()), "adamw", spec)
    f = A * 1150 ** -B
    assert pred[10] == pytest.approx(f)
    assert pred[20] == pytest.approx(f * 0.95)
    assert pred[30] == pytest.approx(f * 0.92)


def test_zero_noise_gives_zero_width_intervals_and_gap_sign():
    fit = loss_fit.run_fit(parsed(exact_rows()), loss_fit.Spec(), n_boot=20)
    t = table(loss_fit.summarize(fit, loss_fit.Spec()))
    for key, r in t.items():
        if key[0] != "p_muon_better":
            assert r["lo"] == pytest.approx(r["value"]) and r["hi"] == pytest.approx(r["value"])
    f = A * 1150 ** -B
    assert t["muon_minus_adamw", "", 30]["value"] == pytest.approx(-0.03 * f * 0.92)
    assert t["gain_from_prev", "adamw", 30]["value"] == pytest.approx(f * (0.95 - 0.92))
    assert t["p_muon_better", "", 20]["value"] == 1.0


def test_seed_noise_widens_the_interval_around_the_point():
    spec = loss_fit.Spec()
    t = table(loss_fit.summarize(loss_fit.run_fit(parsed(exact_rows(sd=0.03)), spec, n_boot=300), spec))
    r = t["loss", "adamw", 30]
    assert r["lo"] < r["value"] < r["hi"]


def test_equal_optimizers_are_not_called_a_win():
    spec = loss_fit.Spec()
    fit = loss_fit.run_fit(parsed(exact_rows(sd=0.05, muon_factor=1.0)), spec, n_boot=400)
    t = table(loss_fit.summarize(fit, spec))
    assert 0.2 < t["p_muon_better", "", 30]["value"] < 0.8


def test_single_seed_point_uses_pooled_sd():
    rows = parsed([row(50, 10, "adamw", 3.4, sd=0.02), row(100, 10, "adamw", 3.1, sd=0.02),
                   row(200, 10, "adamw", 2.9, sd=None, n=1)])
    assert loss_fit.pooled_sd(rows, "adamw") == pytest.approx(0.02)
    assert loss_fit.std_error(rows[2], 0.02) == pytest.approx(0.02)
    assert loss_fit.std_error(rows[0], 0.02) == pytest.approx(0.02 / 3 ** 0.5)
    assert loss_fit.pooled_sd(rows[2:], "adamw") == 0.0


def test_duplicate_rows_are_refused():
    rows = parsed(exact_rows() + [row(50, 10, "adamw", 3.0)])
    with pytest.raises(ValueError, match="more than one"):
        loss_fit.predict_optimizer(rows, "adamw", loss_fit.Spec())


def test_missing_token_budget_is_refused():
    rows = parsed(r for r in exact_rows() if not (r["tok_per_param"] == 30 and r["optimizer"] == "muon"))
    with pytest.raises(ValueError, match="muon: no size has wsd rows at both 10 and 30"):
        loss_fit.run_fit(rows, loss_fit.Spec(), n_boot=2)


def test_one_size_is_not_a_trend():
    rows = parsed(r for r in exact_rows() if r["schedule"] != "cosine" or r["params_m"] == 50)
    with pytest.raises(ValueError, match="two or more sizes"):
        loss_fit.predict_optimizer(rows, "adamw", loss_fit.Spec())


def test_dtype_filter_keeps_precisions_apart(tmp_path):
    path = str(tmp_path / "best.csv")
    write_best(path, exact_rows() + [row(50, 10, "adamw", 9.9, dtype="bfloat16")])
    assert all(r["mean"] < 9 for r in loss_fit.load_best(path))
    assert any(r["mean"] > 9 for r in loss_fit.load_best(path, dtype=None))


def test_backtest_is_exact_on_a_power_law_and_needs_three_sizes():
    bt = loss_fit.backtest(parsed(exact_rows()), "adamw", loss_fit.Spec())
    assert bt["size_m"] == 350 and bt["error"] == pytest.approx(0.0, abs=1e-9)
    few = parsed(r for r in exact_rows() if r["schedule"] != "cosine" or r["params_m"] in (50, 100))
    assert loss_fit.backtest(few, "adamw", loss_fit.Spec()) is None


def test_e1_numbers_from_the_learning_rate_sheet():
    """Final losses at 10 tok/param (50M, 100M, 200M) and 20 tok/param (50M) from Team3_Learning_Rates.xlsx."""
    rows = parsed([row(50, 10, "adamw", 3.4119), row(100, 10, "adamw", 3.1623), row(200, 10, "adamw", 2.9495, None, 1),
                   row(50, 10, "muon", 3.2990), row(100, 10, "muon", 3.0614), row(200, 10, "muon", 2.8496, None, 1),
                   row(50, 20, "adamw", 3.2453), row(50, 20, "muon", 3.1735)])
    spec = loss_fit.Spec(tpps=(20,), token_schedule="cosine")
    pred = {o: loss_fit.predict_optimizer(rows, o, spec) for o in loss_fit.OPTIMIZERS}
    for o, last in (("adamw", 2.9495), ("muon", 2.8496)):
        assert pred[o][10] < last and pred[o][20] < pred[o][10] # bigger and longer both help
    assert pred["adamw"][20] / pred["adamw"][10] == pytest.approx(3.2453 / 3.4119)
    assert pred["muon"][10] < pred["adamw"][10]


def test_cli_prints_and_writes_the_table(tmp_path, capsys):
    best, out = str(tmp_path / "best.csv"), str(tmp_path / "fit.csv")
    write_best(best, exact_rows(sd=0.02))
    loss_fit.main([best, "--n-boot", "50", "--out", out])
    printed = capsys.readouterr().out
    assert "backtest" in printed and "Muon is ahead" in printed
    got = list(csv.DictReader(open(out)))
    assert {r["quantity"] for r in got} == {"loss", "gain_from_prev", "muon_minus_adamw", "p_muon_better"}
    adamw_loss = [r for r in got if r["quantity"] == "loss" and r["optimizer"] == "adamw"]
    assert [r["tok_per_param"] for r in adamw_loss] == ["10", "20", "30"]
