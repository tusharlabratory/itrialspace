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
Generate the headline VLM evaluation report on a fixed case split.

Computes accuracy for every (model x condition x task) **restricted to one frozen
split**, so every number in the report shares a single N (the reporting-N
convention; see ``docs/vlm_eval.md``). Emits a tidy CSV and a markdown report.

Example
-------
python -m itrialspace.evaluation.vlm_eval.report \\
    --data-base $ITRIALSPACE_DATA_DIR/vlm_dataset \\
    --split release_v1_full \\
    --out docs/vlm_results.md \\
    --out-csv $ITRIALSPACE_DATA_DIR/vlm_dataset/vlm_accuracy_release_v1_full.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from itrialspace.evaluation.vlm_eval.splits import canonical_uid, load_split

MODELS = [("BiomedCLIP", "biomedclip"), ("LLaVA-Med", "llava_med"), ("MedGemma", "medgemma")]
PROFILE = {"biomedclip": "lung_axial", "llava_med": "lung_axial", "medgemma": "lung_axial_medgemma"}
CONDS = ["plain", "bbox", "contour", "bbox_contour"]
TASKS = ["presence", "lobe", "size"]
CHANCE = {"presence": 0.50, "lobe": 0.20, "size": 0.25}


def _accuracy(csv_path: str, keep: set) -> Tuple[Optional[float], int, int]:
    """Accuracy over rows whose canonical uid is in ``keep`` (deduped by uid)."""
    seen: Dict[str, bool] = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            ip = r.get("image_path") or r.get("png_path") or ""
            u = canonical_uid(ip)
            if u in keep:
                seen[u] = str(r.get("correct", "")).lower() in ("true", "1")
    if not seen:
        return None, 0, 0
    n_correct = sum(seen.values())
    return n_correct / len(seen), len(seen), n_correct


def build_rows(data_base: str, split: str, set_name: str) -> Tuple[int, List[dict]]:
    keep = load_split(f"{data_base}/splits/{split}.{set_name}.txt")
    rows: List[dict] = []
    for disp, m in MODELS:
        base = f"{data_base}/{set_name}/{PROFILE[m]}/{m}"
        for t in TASKS:
            for c in CONDS:
                fp = f"{base}/{c}/{t}_results.csv"
                acc, n, nc = _accuracy(fp, keep) if os.path.isfile(fp) else (None, 0, 0)
                rows.append(
                    {
                        "set": set_name,
                        "model": disp,
                        "task": t,
                        "condition": c,
                        "accuracy": "" if acc is None else round(acc, 4),
                        "n": n,
                        "n_correct": nc,
                    }
                )
    return len(keep), rows


def _md_table(rows: List[dict]) -> str:
    by: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in rows:
        by.setdefault((r["model"], r["task"]), {})[r["condition"]] = r["accuracy"]
    out = [
        "| Model | Task | plain | bbox | contour | bbox_contour | best |",
        "|-------|------|------:|-----:|--------:|-------------:|------|",
    ]
    for disp, _ in MODELS:
        for t in TASKS:
            cells = by.get((disp, t), {})
            vals = {c: cells.get(c) for c in CONDS}
            fmt = {c: (f"{vals[c]*100:.1f}" if isinstance(vals[c], float) else "–") for c in CONDS}
            numeric = {c: v for c, v in vals.items() if isinstance(v, float)}
            best = max(numeric, key=numeric.get) if numeric else "–"
            out.append(
                f"| {disp} | {t} | {fmt['plain']} | {fmt['bbox']} | {fmt['contour']} "
                f"| {fmt['bbox_contour']} | {best} |"
            )
    return "\n".join(out)


def write_report(data_base: str, split: str, out_md: str, out_csv: Optional[str]) -> None:
    sections, all_rows, ns = [], [], {}
    for set_name in ("synthetic", "real"):
        n, rows = build_rows(data_base, split, set_name)
        ns[set_name] = n
        all_rows.extend(rows)
        sections.append((set_name, n, rows))

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["set", "model", "task", "condition", "accuracy", "n", "n_correct"]
            )
            w.writeheader()
            w.writerows(all_rows)

    lines = [
        "# VLM evaluation results",
        "",
        f"> Generated {date.today().isoformat()} from `{split}` · accuracy (%) · "
        "three zero-shot tasks × four image conditions × three models.",
        "",
        "**Reporting set.** Every number below is computed on the **same fixed cases** — the "
        f"`{split}` split: only cases that have **all four image conditions** (plain, bbox, "
        "contour, bbox_contour), so every model/condition/task is scored on an identical N and "
        "the columns are directly comparable. Accuracy = mean(`prediction == ground_truth`); "
        "the case key is the canonical slice uid (not the `case_id` index).",
        "",
        f"- **Synthetic:** N = **{ns['synthetic']:,}** cases · **Real:** N = **{ns['real']:,}** cases",
        "- Chance: presence 50% · lobe (5-class) 20% · size (4-class) 25%",
        "",
    ]
    for set_name, n, rows in sections:
        lines += [f"## {set_name.capitalize()} (N = {n:,})", "", _md_table(rows), ""]
    lines += [
        "## Notes",
        "",
        "- Split, checksums, and provenance: `vlm_dataset/splits/` (`splits.json`).",
        "- Reproduce: `python -m itrialspace.evaluation.vlm_eval.report --data-base "
        f"$ITRIALSPACE_DATA_DIR/vlm_dataset --split {split} --out {out_md}`.",
        "- The conference submission reported synthetic N = 42,382 on the plain set; this report "
        "uses the all-conditions split for strict apples-to-apples condition comparison.",
        "",
    ]
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines))
    print(f"[report] {out_md}  (synthetic N={ns['synthetic']:,}, real N={ns['real']:,})")
    if out_csv:
        print(f"[report] {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="VLM evaluation report on a fixed split.")
    ap.add_argument("--data-base", required=True, help="Path to vlm_dataset/.")
    ap.add_argument(
        "--split", default="release_v1_full", help="Split name (default: release_v1_full)."
    )
    ap.add_argument("--out", required=True, help="Output markdown report path.")
    ap.add_argument("--out-csv", default=None, help="Optional tidy CSV path.")
    args = ap.parse_args()
    write_report(args.data_base, args.split, args.out, args.out_csv)


if __name__ == "__main__":
    main()
