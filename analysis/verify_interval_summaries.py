"""Independently reconcile uploaded aggregate tables; no row-level claims."""

import argparse
import csv
import json
import math
from pathlib import Path


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify(directory):
    directory = Path(directory)
    overall = {k: float(v) for k, v in read_csv(directory/"interval_metrics.csv")[0].items()}
    n = overall["n_observations"]
    metrics = [name for name in overall if name not in ["n_observations", "nominal_target_coverage"]]
    groups, checks = {}, []
    for filename, dimension in [("interval_by_band.csv", "horizon_band"),
                                ("interval_by_sales_state.csv", "sales_state")]:
        rows = read_csv(directory/filename)
        assert sum(float(row["n_observations"]) for row in rows) == n, filename
        for metric in metrics:
            weighted = sum(float(row[metric])*float(row["n_observations"]) for row in rows)/n
            assert math.isclose(weighted, overall[metric], rel_tol=1e-10, abs_tol=1e-12), (filename, metric)
        groups[dimension] = []
        for row in rows:
            count, rate = int(row["n_observations"]), float(row["coverage"])
            assert math.isclose(count*rate, round(count*rate), abs_tol=1e-8)
            groups[dimension].append({dimension: row[dimension], "n": count,
                                      "covered_count": round(count*rate), "coverage": rate,
                                      "mean_width": float(row["mean_width"]),
                                      "interval_score_80_from_pinball": 10*(float(row["pinball_q10"])+float(row["pinball_q90"]))})
        checks.append(f"{filename}: counts and weighted summaries reconcile")
    assert math.isclose(overall["mean_width"], overall["mean_upper"]-overall["mean_lower"], abs_tol=1e-12)
    for name in ["coverage", "coverage_raw", "raw_crossing_rate", "median_outside_raw_bounds_rate", "bounds_adjustment_rate"]:
        assert 0 <= overall[name] <= 1, name
    assert overall["raw_crossing_rate"] == overall["bounds_adjustment_rate"] == 0
    assert overall["coverage_raw"] == overall["coverage"]
    receipt = json.loads((directory/"run.json").read_text(encoding="utf-8"))
    assert receipt["stage"] == "validation" and receipt["holdout_evaluated"] is False
    assert receipt["control_reproduced_within_tolerance"] is True
    assert n == receipt["n_series"]*receipt["config"]["horizon"]*receipt["config"]["n_windows"]
    points = read_csv(directory/"point_overall_metrics.csv")
    totals = [float(row["mae"])*float(row["n_observations"])/float(row["wape"]) for row in points]
    assert all(math.isclose(t, totals[0], rel_tol=1e-10) for t in totals)
    assert all(int(row["n_observations"]) == n for row in points)
    return {"source_sha256": receipt["source_sha256"], "n": int(n),
            "covered_count": round(n*overall["coverage"]), "coverage": overall["coverage"],
            "mean_width": overall["mean_width"], "target_coverage": overall["nominal_target_coverage"],
            "coverage_gap_percentage_points": 100*(overall["coverage"]-overall["nominal_target_coverage"]),
            "interval_score_80_from_pinball": 10*(overall["pinball_q10"]+overall["pinball_q90"]),
            "interval_score_identity": "IS80 = 10*(pinball_q10 + pinball_q90); valid here because raw ordered tails equal delivered bounds",
            "groups": groups, "implied_actual_total_from_point_summaries": totals[0],
            "checks": checks, "scope": "Summary arithmetic and receipt only; interval row-level predictions not supplied"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", default="reports/interval_reference")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = json.dumps(verify(args.reference_dir), indent=2)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    print(result)
