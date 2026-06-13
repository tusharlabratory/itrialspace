#!/usr/bin/env python3
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
iTrialSpace Trial Runner — builds a cohort manifest from a trial mode.

Usage:
    python run_trial.py --mode <MODE> [OPTIONS]

Modes:
    1  controlled_prevalence    Fixed malignancy prevalence
    2  size_detection_curve     One cohort per size bucket (FROC)
    3  location_sensitivity     One cohort per lobe
    4  demographic_strat        Cohorts by demographic strata
    5  counterfactual            Same cohort, one parameter varied
    6  cross_dataset             One cohort per dataset as sole donor
    7  bootstrap_confidence     N bootstrap replicates of a base cohort
    8  algorithm_comparison     Single standardised cohort for comparing models
    9  screening_simulation     Multi-round screening protocol
   10  multi_nodule             Single vs multi-nodule sub-cohorts
   11  digital_twin_isolation   Per-nodule isolation within host anatomy
   12  digital_twin_complete    Full multi-nodule digital twin per CT scan
   13  digital_twin_cross       Cross-patient host anatomy + donor nodules

All manifests are written to --output-dir (default: ./output/).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Add itrialspace to path
ITRIALSPACE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ITRIALSPACE_ROOT)


def load_index():
    """Load the full NoduleIndex from the dataset registry.

    The dataset registry config is discovered via ``itrialspace.config.settings``
    (configs/datasets.yaml → datasets.example.yaml → bundled default), with paths
    resolved from ``$ITRIALSPACE_DATA_DIR``.
    """
    from itrialspace import DatasetRegistry, NoduleIndex

    registry = DatasetRegistry.from_yaml()
    print("Loading index from configured dataset registry ...")
    idx = NoduleIndex.from_registry(registry, verbose=True)
    print(f"Loaded {len(idx.df):,} nodules from {idx.df['dataset'].nunique()} datasets")
    return idx


def build_and_save(builder, specs, output_dir, verify_paths=True, args=None):
    """Build manifests from specs and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    if not isinstance(specs, list):
        specs = [specs]

    for i, spec in enumerate(specs):
        print(f"\n{'='*60}")
        print(f"Building: {spec.trial_name} ({i+1}/{len(specs)})")
        print(f"  n_cases={spec.n_cases}, mode={spec.cohort_mode}, seed={spec.seed}")
        if spec.malignancy_prevalence is not None:
            print(f"  target prevalence={spec.malignancy_prevalence:.3f}")
        print(f"{'='*60}")

        t0 = time.time()

        if spec.n_bootstrap > 1:
            manifests = builder.build_all(spec, verbose=True)
            for j, m in enumerate(manifests):
                name = f"{spec.trial_name}_boot{j:03d}"
                csv_path = os.path.join(output_dir, f"{name}.csv")
                m.to_csv(csv_path)
                print(f"  Bootstrap {j}: {len(m)} cases → {csv_path}")
            manifest = manifests[0]  # use first for summary
        else:
            manifest = builder.build(spec, verbose=True)
            csv_path = os.path.join(output_dir, f"{spec.trial_name}.csv")
            json_path = os.path.join(output_dir, f"{spec.trial_name}.json")
            manifest.to_csv(csv_path)
            manifest.to_json(json_path)
            print(f"  Saved CSV:  {csv_path}")
            print(f"  Saved JSON: {json_path}")

        elapsed = time.time() - t0
        print(f"\n  Built in {elapsed:.1f}s")

        # Summary
        summary = manifest.summary()
        print(f"  Cases: {summary['n_cases']}")
        if "n_malignant" in summary:
            print(f"  Malignant: {summary['n_malignant']}, Benign: {summary['n_benign']}")
        if "malignancy_rate" in summary:
            print(f"  Actual prevalence: {summary['malignancy_rate']:.4f}")
        if "donor_datasets" in summary:
            print(f"  Donor datasets: {summary['donor_datasets']}")

        # Audit
        audit = manifest.audit()
        if len(audit) > 0:
            print("\n  Audit:")
            for _, row in audit.iterrows():
                print(
                    f"    {row['metric']}: target={row['target']}, actual={row['actual']}, dev={row['deviation']}"
                )

        # Path verification
        if verify_paths:
            missing = manifest.verify_paths()
            n_total = len(manifest) * 4
            n_found = n_total - len(missing)
            print(f"\n  Paths: {n_found}/{n_total} exist ({n_found/n_total*100:.1f}%)")
            if len(missing) > 0:
                by_type = missing["path_type"].value_counts().to_dict()
                print(f"  Missing by type: {by_type}")

    # Save run config
    config_path = os.path.join(output_dir, "_run_config.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "n_specs": len(specs),
                "trial_names": [s.trial_name for s in specs],
                "args": vars(args),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nRun config saved to {config_path}")


def run(args):
    from itrialspace.site.cohort_builder import CohortBuilder
    from itrialspace.site.path_resolver import PathResolver

    idx = load_index()
    builder = CohortBuilder(idx, PathResolver())

    mode = args.mode

    # ── Mode 1: Controlled Prevalence ────────────────────────────────────
    if mode == "controlled_prevalence":
        from itrialspace.site.trial_modes import controlled_prevalence_study

        spec = controlled_prevalence_study(
            n_cases=args.n_cases,
            prevalence=args.prevalence,
            template=args.template,
            seed=args.seed,
        )
        build_and_save(builder, spec, args.output_dir, args.verify_paths, args)

    # ── Mode 2: Size Detection Curve ─────────────────────────────────────
    elif mode == "size_detection_curve":
        from itrialspace.site.trial_modes import size_detection_curve

        specs = size_detection_curve(
            n_per_bucket=args.n_per_bucket,
            label=args.label,
            template=args.template,
            seed=args.seed,
        )
        build_and_save(builder, specs, args.output_dir, args.verify_paths, args)

    # ── Mode 3: Location Sensitivity ─────────────────────────────────────
    elif mode == "location_sensitivity":
        from itrialspace.site.trial_modes import location_sensitivity

        specs = location_sensitivity(
            n_per_lobe=args.n_per_lobe,
            label=args.label,
            diameter_range=(args.diam_min, args.diam_max),
            template=args.template,
            seed=args.seed,
        )
        build_and_save(builder, specs, args.output_dir, args.verify_paths, args)

    # ── Mode 4: Demographic Stratification ───────────────────────────────
    elif mode == "demographic_strat":
        from itrialspace.site.trial_modes import demographic_stratification

        specs = demographic_stratification(
            n_per_stratum=args.n_per_stratum,
            template=args.template,
            seed=args.seed,
        )
        build_and_save(builder, specs, args.output_dir, args.verify_paths, args)

    # ── Mode 5: Counterfactual Cohort ────────────────────────────────────
    elif mode == "counterfactual":
        from itrialspace.site.trial_modes import counterfactual_cohort

        values = [float(v) for v in args.values.split(",")] if args.values else None
        specs = counterfactual_cohort(
            n_cases=args.n_cases,
            template=args.template,
            vary_param=args.vary_param,
            values=values,
            seed=args.seed,
        )
        build_and_save(builder, specs, args.output_dir, args.verify_paths, args)

    # ── Mode 6: Cross-Dataset Generalization ─────────────────────────────
    elif mode == "cross_dataset":
        from itrialspace.site.trial_modes import cross_dataset_generalization

        datasets = args.datasets.split(",") if args.datasets else None
        specs = cross_dataset_generalization(
            n_cases=args.n_cases,
            datasets=datasets,
            seed=args.seed,
        )
        build_and_save(builder, specs, args.output_dir, args.verify_paths, args)

    # ── Mode 7: Bootstrap Confidence ─────────────────────────────────────
    elif mode == "bootstrap_confidence":
        from itrialspace.site.trial_modes import bootstrap_confidence, controlled_prevalence_study

        base = controlled_prevalence_study(
            n_cases=args.n_cases,
            prevalence=args.prevalence,
            template=args.template,
            seed=args.seed,
        )
        spec = bootstrap_confidence(base, n_bootstrap=args.n_bootstrap)
        build_and_save(builder, spec, args.output_dir, args.verify_paths, args)

    # ── Mode 8: Algorithm Comparison ─────────────────────────────────────
    elif mode == "algorithm_comparison":
        from itrialspace.site.trial_modes import algorithm_comparison

        spec = algorithm_comparison(
            n_cases=args.n_cases,
            template=args.template,
            seed=args.seed,
        )
        build_and_save(builder, spec, args.output_dir, args.verify_paths, args)

    # ── Mode 9: Screening Protocol Simulation ────────────────────────────
    elif mode == "screening_simulation":
        from itrialspace.site.trial_modes import screening_protocol_simulation

        specs = screening_protocol_simulation(
            template=args.template,
            n_rounds=args.n_rounds,
            n_cases_per_round=args.n_cases_per_round,
            prevalence_decay=args.prevalence_decay,
            seed=args.seed,
        )
        build_and_save(builder, specs, args.output_dir, args.verify_paths, args)

    # ── Mode 10: Multi-Nodule Realism Study ──────────────────────────────
    elif mode == "multi_nodule":
        from itrialspace.site.trial_modes import multi_nodule_realism_study

        specs = multi_nodule_realism_study(
            n_cases=args.n_cases,
            multi_nodule_fraction=args.multi_nodule_fraction,
            companion_strategy=args.companion_strategy,
            template=args.template,
            seed=args.seed,
        )
        build_and_save(builder, specs, args.output_dir, args.verify_paths, args)

    # ── Mode 11: Digital Twin Isolation ────────────────────────────────
    elif mode == "digital_twin_isolation":
        from itrialspace.site.trial_modes import digital_twin_isolation

        if not args.dataset:
            print("ERROR: --dataset is required for digital_twin_isolation mode")
            sys.exit(1)

        iso_spec = digital_twin_isolation(
            dataset=args.dataset,
            max_patients=args.max_patients,
            all_patients=args.all_patients,
            max_nodules_per_patient=args.max_nodules_per_patient,
            label=args.iso_label,
            diameter_min=args.diameter_min,
            diameter_max=args.diameter_max,
            seed=args.seed,
        )

        print(f"\n{'='*60}")
        print(f"Building: {iso_spec.trial_name}")
        print(f"  dataset={iso_spec.dataset}, seed={iso_spec.seed}")
        if iso_spec.max_patients:
            print(f"  max_patients={iso_spec.max_patients}")
        else:
            print("  all_patients=True")
        print(f"{'='*60}")

        t0 = time.time()
        manifest = builder.build_digital_twin_isolation(iso_spec, verbose=True)
        elapsed = time.time() - t0

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{iso_spec.trial_name}.csv")
        json_path = os.path.join(output_dir, f"{iso_spec.trial_name}.json")
        manifest.to_csv(csv_path)
        manifest.to_json(json_path)

        print(f"\n  Built in {elapsed:.1f}s")
        print(f"  Saved CSV:  {csv_path}")
        print(f"  Saved JSON: {json_path}")

        summary = manifest.summary()
        print(f"  Cases: {summary['n_cases']}")
        if "n_malignant" in summary:
            print(f"  Malignant: {summary['n_malignant']}, Benign: {summary['n_benign']}")

        if args.verify_paths:
            missing = manifest.verify_paths()
            n_total = len(manifest) * 4
            n_found = n_total - len(missing)
            print(f"\n  Paths: {n_found}/{n_total} exist ({n_found/n_total*100:.1f}%)")
            if len(missing) > 0:
                by_type = missing["path_type"].value_counts().to_dict()
                print(f"  Missing by type: {by_type}")

        # Save run config
        config_path = os.path.join(output_dir, "_run_config.json")
        with open(config_path, "w") as f:
            json.dump(
                {
                    "mode": mode,
                    "dataset": iso_spec.dataset,
                    "trial_name": iso_spec.trial_name,
                    "args": vars(args),
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\nRun config saved to {config_path}")

    # ── Mode 12: Digital Twin Complete ─────────────────────────────────
    elif mode == "digital_twin_complete":
        from itrialspace.site.trial_modes import digital_twin_complete

        if not args.dataset:
            print("ERROR: --dataset is required for digital_twin_complete mode")
            sys.exit(1)

        complete_spec = digital_twin_complete(
            dataset=args.dataset,
            max_patients=args.max_patients,
            all_patients=args.all_patients,
            label=args.iso_label,
            diameter_min=args.diameter_min,
            diameter_max=args.diameter_max,
            seed=args.seed,
        )

        print(f"\n{'='*60}")
        print(f"Building: {complete_spec.trial_name}")
        print(f"  dataset={complete_spec.dataset}, seed={complete_spec.seed}")
        if complete_spec.max_patients:
            print(f"  max_patients={complete_spec.max_patients}")
        else:
            print("  all_patients=True")
        print(f"{'='*60}")

        t0 = time.time()
        manifest = builder.build_digital_twin_complete(complete_spec, verbose=True)
        elapsed = time.time() - t0

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{complete_spec.trial_name}.csv")
        json_path = os.path.join(output_dir, f"{complete_spec.trial_name}.json")
        manifest.to_csv(csv_path)
        manifest.to_json(json_path)

        print(f"\n  Built in {elapsed:.1f}s")
        print(f"  Saved CSV:  {csv_path}")
        print(f"  Saved JSON: {json_path}")

        summary = manifest.summary()
        print(f"  Cases: {summary['n_cases']}")
        if "n_malignant" in summary:
            print(f"  Malignant: {summary['n_malignant']}, Benign: {summary['n_benign']}")

        if args.verify_paths:
            missing = manifest.verify_paths()
            n_total = len(manifest) * 4
            n_found = n_total - len(missing)
            print(f"\n  Paths: {n_found}/{n_total} exist ({n_found/n_total*100:.1f}%)")
            if len(missing) > 0:
                by_type = missing["path_type"].value_counts().to_dict()
                print(f"  Missing by type: {by_type}")

        # Save run config
        config_path = os.path.join(output_dir, "_run_config.json")
        with open(config_path, "w") as f:
            json.dump(
                {
                    "mode": mode,
                    "dataset": complete_spec.dataset,
                    "trial_name": complete_spec.trial_name,
                    "args": vars(args),
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\nRun config saved to {config_path}")

    # ── Mode 13: Digital Twin Cross ────────────────────────────────────
    elif mode == "digital_twin_cross":
        from itrialspace.site.trial_modes import digital_twin_cross

        if not args.host_dataset:
            print("ERROR: --host-dataset is required for digital_twin_cross mode")
            sys.exit(1)
        if not args.donor_dataset:
            print("ERROR: --donor-dataset is required for digital_twin_cross mode")
            sys.exit(1)

        cross_spec = digital_twin_cross(
            host_dataset=args.host_dataset,
            donor_dataset=args.donor_dataset,
            max_host_patients=args.max_host_patients,
            all_host_patients=args.all_host_patients,
            max_donor_patients=args.max_donor_patients,
            all_donor_patients=args.all_donor_patients,
            max_donor_nodules=args.max_donor_nodules,
            donor_transfer_mode=args.donor_transfer_mode,
            pairing_policy=args.pairing_policy,
            n_hosts_per_donor=args.n_hosts_per_donor,
            placement_strategy=args.placement_strategy,
            label=args.iso_label,
            diameter_min=args.diameter_min,
            diameter_max=args.diameter_max,
            seed=args.seed,
        )

        print(f"\n{'='*60}")
        print(f"Building: {cross_spec.trial_name}")
        print(f"  host_dataset={cross_spec.host_dataset}, donor_dataset={cross_spec.donor_dataset}")
        print(f"  donor_transfer_mode={cross_spec.donor_transfer_mode}")
        print(f"  pairing_policy={cross_spec.pairing_policy}")
        print(f"  placement_strategy={cross_spec.placement_strategy}")
        print(f"  seed={cross_spec.seed}")
        print(f"{'='*60}")

        t0 = time.time()
        manifest = builder.build_digital_twin_cross(cross_spec, verbose=True)
        elapsed = time.time() - t0

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{cross_spec.trial_name}.csv")
        json_path = os.path.join(output_dir, f"{cross_spec.trial_name}.json")
        manifest.to_csv(csv_path)
        manifest.to_json(json_path)

        print(f"\n  Built in {elapsed:.1f}s")
        print(f"  Saved CSV:  {csv_path}")
        print(f"  Saved JSON: {json_path}")

        summary = manifest.summary()
        print(f"  Cases: {summary['n_cases']}")
        if "n_malignant" in summary:
            print(f"  Malignant: {summary['n_malignant']}, Benign: {summary['n_benign']}")

        if args.verify_paths:
            missing = manifest.verify_paths()
            n_total = len(manifest) * 4
            n_found = n_total - len(missing)
            print(f"\n  Paths: {n_found}/{n_total} exist ({n_found/n_total*100:.1f}%)")
            if len(missing) > 0:
                by_type = missing["path_type"].value_counts().to_dict()
                print(f"  Missing by type: {by_type}")

        # Save run config
        config_path = os.path.join(output_dir, "_run_config.json")
        with open(config_path, "w") as f:
            json.dump(
                {
                    "mode": mode,
                    "host_dataset": cross_spec.host_dataset,
                    "donor_dataset": cross_spec.donor_dataset,
                    "trial_name": cross_spec.trial_name,
                    "args": vars(args),
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\nRun config saved to {config_path}")

    else:
        print(f"Unknown mode: {mode}")
        print(
            "Available modes: controlled_prevalence, size_detection_curve, "
            "location_sensitivity, demographic_strat, counterfactual, "
            "cross_dataset, bootstrap_confidence, algorithm_comparison, "
            "screening_simulation, multi_nodule, digital_twin_isolation, "
            "digital_twin_complete, digital_twin_cross"
        )
        sys.exit(1)

    print("\nDone!")


def build_parser():
    parser = argparse.ArgumentParser(
        description="iTrialSpace Trial Runner — build cohort manifests from trial modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", required=True, help="Trial mode name")
    parser.add_argument(
        "--output-dir", default="./output", help="Output directory (default: ./output)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--template", default="NLST", help="Trial template (default: NLST)")
    parser.add_argument(
        "--verify-paths",
        action="store_true",
        default=True,
        help="Verify file paths exist (default: True)",
    )
    parser.add_argument("--no-verify-paths", dest="verify_paths", action="store_false")

    # Mode-specific arguments
    parser.add_argument("--n-cases", type=int, default=500, help="Number of cases")
    parser.add_argument("--prevalence", type=float, default=0.05, help="Malignancy prevalence")
    parser.add_argument("--label", type=int, default=1, help="Label filter (0=benign, 1=malignant)")
    parser.add_argument(
        "--n-per-bucket", type=int, default=100, help="Cases per size bucket (mode 2)"
    )
    parser.add_argument("--n-per-lobe", type=int, default=100, help="Cases per lobe (mode 3)")
    parser.add_argument("--n-per-stratum", type=int, default=200, help="Cases per stratum (mode 4)")
    parser.add_argument("--diam-min", type=float, default=6.0, help="Min diameter mm (mode 3)")
    parser.add_argument("--diam-max", type=float, default=15.0, help="Max diameter mm (mode 3)")
    parser.add_argument(
        "--vary-param", default="malignancy_prevalence", help="Parameter to vary (mode 5)"
    )
    parser.add_argument("--values", default=None, help="Comma-separated values to vary (mode 5)")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names (mode 6)")
    parser.add_argument(
        "--n-bootstrap", type=int, default=100, help="Bootstrap replicates (mode 7)"
    )
    parser.add_argument("--n-rounds", type=int, default=3, help="Screening rounds (mode 9)")
    parser.add_argument(
        "--n-cases-per-round", type=int, default=500, help="Cases per round (mode 9)"
    )
    parser.add_argument(
        "--prevalence-decay", type=float, default=0.7, help="Prevalence decay (mode 9)"
    )
    parser.add_argument(
        "--multi-nodule-fraction",
        type=float,
        default=0.25,
        help="Fraction of multi-nodule cases (mode 10, default: 0.25)",
    )
    parser.add_argument(
        "--companion-strategy",
        default="all_companions",
        help="Companion strategy (mode 10, default: all_companions)",
    )

    # Mode 11/12: Digital twin arguments
    parser.add_argument(
        "--dataset", default=None, help="Source dataset name (mode 11/12, e.g. DLCS24)"
    )
    iso_group = parser.add_mutually_exclusive_group()
    iso_group.add_argument(
        "--max-patients", type=int, default=None, help="Max host patients to select (mode 11/12)"
    )
    iso_group.add_argument(
        "--all-patients",
        action="store_true",
        default=False,
        help="Use all eligible patients (mode 11/12)",
    )
    parser.add_argument(
        "--max-nodules-per-patient",
        type=int,
        default=None,
        help="Cap isolation cases per patient (mode 11)",
    )
    parser.add_argument(
        "--iso-label",
        type=int,
        default=None,
        help="Filter nodules by label (mode 11/12/13, 0=benign, 1=malignant, None=any)",
    )
    parser.add_argument(
        "--diameter-min", type=float, default=None, help="Min nodule diameter mm (mode 11/12/13)"
    )
    parser.add_argument(
        "--diameter-max", type=float, default=None, help="Max nodule diameter mm (mode 11/12/13)"
    )

    # Mode 13: Digital twin cross arguments
    parser.add_argument(
        "--host-dataset", default=None, help="Host anatomy dataset (mode 13, e.g. DLCS24)"
    )
    parser.add_argument(
        "--donor-dataset", default=None, help="Donor nodule dataset (mode 13, e.g. LUNA25)"
    )
    cross_host_group = parser.add_mutually_exclusive_group()
    cross_host_group.add_argument(
        "--max-host-patients", type=int, default=None, help="Max host patients (mode 13)"
    )
    cross_host_group.add_argument(
        "--all-host-patients",
        action="store_true",
        default=False,
        help="Use all eligible host patients (mode 13)",
    )
    cross_donor_group = parser.add_mutually_exclusive_group()
    cross_donor_group.add_argument(
        "--max-donor-patients", type=int, default=None, help="Max donor patients (mode 13)"
    )
    cross_donor_group.add_argument(
        "--all-donor-patients",
        action="store_true",
        default=False,
        help="Use all eligible donor patients (mode 13)",
    )
    parser.add_argument(
        "--max-donor-nodules", type=int, default=None, help="Max total donor nodules (mode 13)"
    )
    parser.add_argument(
        "--donor-transfer-mode",
        default="single",
        choices=["single", "complete"],
        help="Donor transfer mode (mode 13, default: single)",
    )
    parser.add_argument(
        "--pairing-policy",
        default="one_to_one",
        choices=["one_to_one", "one_to_many_hosts", "donor_patient_complete"],
        help="Pairing policy (mode 13, default: one_to_one)",
    )
    parser.add_argument(
        "--n-hosts-per-donor",
        type=int,
        default=3,
        help="Hosts per donor for one_to_many_hosts (mode 13, default: 3)",
    )
    parser.add_argument(
        "--placement-strategy",
        default="profile_faithful_transfer",
        choices=["profile_faithful_transfer", "host_constrained_transfer"],
        help="Placement strategy (mode 13, default: profile_faithful_transfer)",
    )
    return parser


def main(argv=None):
    """Console-script entry point (``its-trials``)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
