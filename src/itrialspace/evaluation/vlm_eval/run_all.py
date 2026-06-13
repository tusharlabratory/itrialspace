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
End-to-end convenience runner: build dataset -> infer -> metrics -> plots.

Supports multiple models via --model flag.

Usage:
    python -m itrialspace.evaluation.vlm_eval.run_all \
        --manifest /path/to/mode1/manifest.csv \
        --manifest /path/to/mode2/manifest.csv \
        --output-dir outputs/vlm_eval \
        --model biomedclip

    python -m itrialspace.evaluation.vlm_eval.run_all \
        --manifest /path/to/mode1/manifest.csv \
        --output-dir outputs/vlm_eval \
        --model llava_med
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

from itrialspace.evaluation.vlm_eval.build_dataset import build_dataset

logger = logging.getLogger(__name__)


SUPPORTED_MODELS = ["biomedclip", "llava_med"]


def run_all(
    manifest_paths: List[str],
    output_dir: str,
    model_name: str = "biomedclip",
    task_names: Optional[List[str]] = None,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    ct_base: str = os.path.join(
        os.environ.get("ITRIALSPACE_OUTPUT_DIR")
        or os.environ.get("ITRIALSPACE_DATA_DIR")
        or os.path.expanduser("~/.itrialspace/data"),
        "generated_cts",
    ),
    mask_base: str = os.path.join(
        os.environ.get("ITRIALSPACE_OUTPUT_DIR")
        or os.environ.get("ITRIALSPACE_DATA_DIR")
        or os.path.expanduser("~/.itrialspace/data"),
        "inserted_masks",
    ),
    max_cases: Optional[int] = None,
    save_overlay: bool = False,
):
    """Run the full pipeline: build dataset, infer, evaluate."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    )

    # Step 1: Build evaluation dataset
    logger.info("STEP 1: Building evaluation dataset")
    dataset_dir = os.path.join(output_dir, "dataset")
    eval_df = build_dataset(
        manifest_paths=manifest_paths,
        output_dir=dataset_dir,
        ct_base=ct_base,
        mask_base=mask_base,
        max_cases=max_cases,
        save_overlay=save_overlay,
    )

    if eval_df.empty:
        logger.error("No cases in eval dataset. Check manifest paths and CT paths.")
        sys.exit(1)

    dataset_csv = os.path.join(dataset_dir, "eval_dataset.csv")

    # Step 2: Run model-specific inference + metrics + plots
    logger.info("STEP 2: Running %s inference", model_name)

    if model_name == "biomedclip":
        from itrialspace.evaluation.vlm_eval.run_biomedclip import run_biomedclip

        results_dir = os.path.join(output_dir, "biomedclip")
        all_metrics = run_biomedclip(
            dataset_csv=dataset_csv,
            output_dir=results_dir,
            task_names=task_names,
            device=device,
            cache_dir=cache_dir,
        )
    elif model_name == "llava_med":
        from itrialspace.evaluation.vlm_eval.runners.run_llava_med import run_llava_med

        results_dir = os.path.join(output_dir, "llava_med")
        all_metrics = run_llava_med(
            dataset_csv=dataset_csv,
            output_dir=results_dir,
            task_names=task_names,
            device=device,
            cache_dir=cache_dir,
            max_cases=max_cases,
        )
    else:
        logger.error("Unknown model '%s'. Supported: %s", model_name, SUPPORTED_MODELS)
        sys.exit(1)

    # Summary
    logger.info("SUMMARY (%s)", model_name)
    if all_metrics:
        for task_name, metrics in all_metrics.items():
            logger.info(
                "  %-12s  accuracy=%.4f  (n=%d)",
                task_name,
                metrics["accuracy"],
                metrics["n_samples"],
            )

    logger.info("All outputs in: %s", output_dir)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    from itrialspace.evaluation.vlm_eval.prompt_templates import ALL_TASKS

    parser = argparse.ArgumentParser(
        description="End-to-end VLM evaluation: build -> infer -> evaluate.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        dest="manifests",
        help="Path to manifest CSV (can be repeated).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root output directory for all artifacts.",
    )
    parser.add_argument(
        "--model",
        default="biomedclip",
        choices=SUPPORTED_MODELS,
        help="Model to run. Default: biomedclip.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=list(ALL_TASKS.keys()),
        help="Task(s) to evaluate. Default: all.",
    )
    parser.add_argument("--device", default=None, help="PyTorch device.")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache dir.")
    parser.add_argument(
        "--ct-base",
        default=os.path.join(
            os.environ.get("ITRIALSPACE_OUTPUT_DIR")
            or os.environ.get("ITRIALSPACE_DATA_DIR")
            or os.path.expanduser("~/.itrialspace/data"),
            "generated_cts",
        ),
    )
    parser.add_argument(
        "--mask-base",
        default=os.path.join(
            os.environ.get("ITRIALSPACE_OUTPUT_DIR")
            or os.environ.get("ITRIALSPACE_DATA_DIR")
            or os.path.expanduser("~/.itrialspace/data"),
            "inserted_masks",
        ),
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit total cases for quick testing.",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        default=False,
        help="Save QC overlay PNGs (lesion mask outline on CT slice).",
    )
    args = parser.parse_args()

    run_all(
        manifest_paths=args.manifests,
        output_dir=args.output_dir,
        model_name=args.model,
        task_names=args.tasks,
        device=args.device,
        cache_dir=args.cache_dir,
        ct_base=args.ct_base,
        mask_base=args.mask_base,
        max_cases=args.max_cases,
        save_overlay=args.save_overlay,
    )


if __name__ == "__main__":
    main()
