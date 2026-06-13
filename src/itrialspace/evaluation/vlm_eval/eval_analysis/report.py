# Copyright (c) 2026 Fakrul Islam Tushar
# Department of Radiology and Imaging Sciences, University of Arizona
# Email: fitushar@arizona.edu
#
# This file is part of iTrialSpace — a virtual clinical trial engine
# for controlled evaluation of lung CT AI models.
#
# If you use this software or the NoduleIndex dataset, please cite:
#
#   @article{tushar2026itrialspace,
#     title   = {iTRIALSPACE: Programmable Virtual Lesion Trials for
#                Controlled Evaluation of Lung CT Models},
#     author  = {Tushar, Fakrul Islam and Momy, Umme Hafsa and
#                Lo, Joseph Y and Rubin, Geoffrey D},
#     journal = {arXiv preprint arXiv:2605.05761},
#     year    = {2026}
#   }
#
# Licensed under the PolyForm Noncommercial License 1.0.0.
# Free to use, copy, modify, and share for NONCOMMERCIAL purposes —
# including academic research and teaching. Commercial use requires
# a separate license.
# Full terms: LICENSE file in the project root, or
# https://polyformproject.org/licenses/noncommercial/1.0.0/
#
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0

"""
Orchestrate the full VLM analysis: one call -> tables/ + figures/ + report.md.

    python -m itrialspace.evaluation.vlm_eval.eval_analysis \\
        --results $ITRIALSPACE_DATA_DIR/vlm_dataset \\
        --split   release_v1_full \\
        --out     analysis_full/

Everything is auto-discovered, so the same command analyses the full dataset or a
small demo run (just point ``--results`` at the demo output dir and omit
``--split``).
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from itrialspace.evaluation.vlm_eval.eval_analysis import load, plots, stats, tables

logger = logging.getLogger(__name__)

_BREAKDOWNS = ["mode", "source_dataset", "lobe", "size_bucket"]
# (by, task) pairs rendered as figures; CSV always holds every condition.
_BREAKDOWN_FIGS = [
    ("source_dataset", "presence"),
    ("source_dataset", "lobe"),
    ("source_dataset", "size"),
    ("size_bucket", "size"),
    ("lobe", "lobe"),
    ("mode", "presence"),
]
_CONF_TASKS = ["lobe", "size"]


def _resolve_split(results: List[str], split: Optional[str]) -> Optional[Dict[str, str]]:
    """Map a split *name* to per-set files under <results>/splits, if present."""
    if not split:
        return None
    if Path(split).is_file():
        return split  # explicit file -> applied to all sets
    for root in results:
        sd = Path(root) / "splits"
        cand = {s: sd / f"{split}.{s}.txt" for s in ("synthetic", "real")}
        found = {s: str(p) for s, p in cand.items() if p.is_file()}
        if found:
            return found
    logger.warning("Split %r not found under any results/splits; using all cases.", split)
    return None


def run(
    results: List[str],
    out: str,
    split: Optional[str] = None,
    n_boot: int = 1000,
    baseline: str = "plain",
) -> None:
    out_dir = Path(out)
    tdir, fdir = out_dir / "tables", out_dir / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    split_arg = _resolve_split(results, split)
    long = load.load_long(results, split=split_arg)
    long.to_parquet(tdir / "long_results.parquet") if _has_parquet() else None

    sets = sorted(long["set"].unique())
    ns = {s: int(long[long["set"] == s]["uid"].nunique()) for s in sets}
    conds_present = [c for c in load.CONDITIONS if c in long["condition"].unique()]

    # ── tables ────────────────────────────────────────────────────────────────
    acc = tables.accuracy_table(long)
    acc.to_csv(tdir / "accuracy.csv", index=False)
    delta = tables.delta_table(long, baseline=baseline)
    delta.to_csv(tdir / "deltas.csv", index=False)
    tables.confusion_long(long).to_csv(tdir / "confusion.csv", index=False)
    tables.per_class_accuracy(long).to_csv(tdir / "per_class_accuracy.csv", index=False)

    bd_all = []
    for by in _BREAKDOWNS:
        bd = tables.breakdown_table(long, by)
        if not bd.empty:
            bd_all.append(bd)
    breakdowns = pd.concat(bd_all, ignore_index=True) if bd_all else pd.DataFrame()
    if not breakdowns.empty:
        breakdowns.to_csv(tdir / "breakdowns.csv", index=False)

    # ── stats ───────────────────────────────────────────────────────────────--
    ci = stats.bootstrap_ci(long, n_boot=n_boot)
    ci.to_csv(tdir / "bootstrap_ci.csv", index=False)
    mc = stats.mcnemar_vs_baseline(long, baseline=baseline)
    if not mc.empty:
        mc.to_csv(tdir / "mcnemar_vs_baseline.csv", index=False)

    # ── figures ───────────────────────────────────────────────────────────────
    fig_index: List[str] = []

    def _log(p):
        if p:
            fig_index.append(str(Path(p).relative_to(out_dir)))

    for s in sets:
        _log(plots.bar_accuracy_by_condition(acc, s, fdir / f"accuracy_by_condition_{s}.png"))
        _log(plots.bar_delta_vs_baseline(delta, s, baseline, fdir / f"delta_vs_{baseline}_{s}.png"))
        if not breakdowns.empty:
            for by, task in _BREAKDOWN_FIGS:
                sl = breakdowns[(breakdowns["group_by"] == by) & (breakdowns["set"] == s)]
                if sl.empty or task not in sl["task"].unique():
                    continue
                _log(
                    plots.bar_breakdown(
                        breakdowns, s, task, baseline, fdir / f"breakdown_{by}_{task}_{s}.png"
                    )
                )
        conf = tables.confusion_long(long)
        for model in sorted(long[long["set"] == s]["model"].unique()):
            for task in _CONF_TASKS:
                if task not in long["task"].unique():
                    continue
                _log(
                    plots.heatmap_confusion(
                        conf,
                        s,
                        model,
                        baseline,
                        task,
                        fdir / f"confusion_{s}_{model}_{task}_{baseline}.png",
                    )
                )

    # ── report.md ───────────────────────────────────────────────────────────--
    _write_markdown(out_dir, split, baseline, sets, ns, conds_present, acc, mc, fig_index, n_boot)
    print(f"[eval_analysis] wrote {out_dir}/report.md  (+ tables/ + {len(fig_index)} figures)")


def _write_markdown(out_dir, split, baseline, sets, ns, conds, acc, mc, fig_index, n_boot):
    L = [
        "# VLM evaluation analysis",
        "",
        f"> Generated {date.today().isoformat()}"
        + (f" · split `{split}`" if split else " · all available cases")
        + f" · baseline `{baseline}`.",
        "",
        "Every number is computed on the same fixed cases per set (one N), keyed by the canonical "
        "slice uid. Accuracy = mean(`prediction == ground_truth`).",
        "",
        "**Cases:** " + " · ".join(f"{s} N={ns[s]:,}" for s in sets),
        "Chance: presence 50% · lobe 20% · size 25%.",
        "",
    ]
    for s in sets:
        L += [
            f"## {s.capitalize()} (N = {ns[s]:,})",
            "",
            tables.accuracy_markdown(acc, s, conds),
            "",
            f"![accuracy](figures/accuracy_by_condition_{s}.png)",
            f"![delta](figures/delta_vs_{baseline}_{s}.png)",
            "",
        ]

    if mc is not None and not mc.empty:
        sig = mc[mc.get("significant_0.05", False)]
        L += [
            "## Significance (McNemar vs baseline)",
            "",
            f"{len(sig)}/{len(mc)} condition effects significant at p<0.05. "
            "Largest accuracy gains (paired):",
            "",
        ]
        top = mc.sort_values("acc_delta", ascending=False).head(8)
        L += ["| set | model | task | condition | Δacc | p |", "|---|---|---|---|---:|---:|"]
        for _, r in top.iterrows():
            L.append(
                f"| {r['set']} | {r['model']} | {r['task']} | {r['condition']} "
                f"| {r['acc_delta']*100:+.1f} | {r['p_value']:.1e} |"
            )
        L.append("")

    L += ["## All figures", ""] + [f"- `{p}`" for p in fig_index] + [""]
    L += [
        "## Outputs",
        "",
        "- `tables/` — accuracy, deltas, breakdowns, confusion, per_class, bootstrap_ci, mcnemar (CSV)",
        "- `figures/` — PNG + PDF",
        f"- Bootstrap: {n_boot} resamples (`tables/bootstrap_ci.csv`).",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join([x for x in L if x is not None]))


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="Full VLM evaluation analysis suite.")
    ap.add_argument("--results", nargs="+", required=True, help="Result root(s) to analyse.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument(
        "--split", default=None, help="Split name (under results/splits) or a file path."
    )
    ap.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples (0 to skip).")
    ap.add_argument("--baseline", default="plain", help="Baseline condition for deltas/McNemar.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(args.results, args.out, split=args.split, n_boot=args.n_boot, baseline=args.baseline)


if __name__ == "__main__":
    main()
