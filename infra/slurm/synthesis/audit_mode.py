#!/usr/bin/env python3
"""Post-run audit for a NodMAISI mode output directory.

Scans all case directories, checks completeness (synthetic_ct, audit, QC),
aggregates per-case results, and writes a mode-level pipeline_summary.json.

Usage:
    python3 audit_mode.py <mode_output_dir> [--mask-dir <mask_dir>]

Example:
    python3 audit_mode.py /scratch/.../generated_cts/mode1_controlled_prevalence \
        --mask-dir /scratch/.../inserted_masks/mode1_controlled_prevalence
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime


def discover_case_dirs(out_dir):
    """Find all case directories (flat and sub-bin layouts)."""
    case_dirs = {}
    for entry in sorted(os.listdir(out_dir)):
        fp = os.path.join(out_dir, entry)
        if not os.path.isdir(fp):
            continue
        if entry.startswith("iTS--"):
            case_dirs[entry] = fp
        else:
            # Sub-bin directory (e.g. size_curve_<5mm/)
            for sub in sorted(os.listdir(fp)):
                sfp = os.path.join(fp, sub)
                if os.path.isdir(sfp) and sub.startswith("iTS--"):
                    case_dirs[sub] = sfp
    return case_dirs


def discover_expected_cases(mask_dir):
    """Discover all expected case IDs from the mask directory."""
    case_ids = set()
    if not mask_dir or not os.path.isdir(mask_dir):
        return case_ids

    # Pass 1: audit.json files
    audited_dirs = set()
    for af in sorted(glob.glob(os.path.join(mask_dir, "**", "audit.json"), recursive=True)):
        audited_dirs.add(os.path.dirname(af))
        try:
            audit = json.load(open(af))
        except Exception:
            continue
        for r in audit.get("records", []):
            if r.get("status") != "success":
                continue
            cp = r.get("output_combined_path", "")
            if not cp or not os.path.isfile(cp):
                continue
            name = os.path.basename(cp)
            for sfx in ("_mask.nii.gz", ".nii.gz"):
                if name.endswith(sfx):
                    name = name[: -len(sfx)]
                    break
            case_ids.add(name)

    # Pass 2: file-scan for dirs without audit.json
    for root, dirs, files in os.walk(mask_dir):
        if root in audited_dirs:
            continue
        for f in files:
            if not f.endswith(".nii.gz"):
                continue
            name = f
            for sfx in ("_mask.nii.gz", ".nii.gz"):
                if name.endswith(sfx):
                    name = name[: -len(sfx)]
                    break
            case_ids.add(name)

    return case_ids


def audit_case(case_id, case_dir):
    """Audit a single case directory for completeness."""
    result = {
        "case_id": case_id,
        "status": "unknown",
        "has_synthetic_ct": False,
        "has_nodmaisi_audit": False,
        "has_qc_slices": False,
        "qc_slice_count": 0,
        "has_failure_json": False,
        "has_dataset_json": False,
        "has_input_mask": False,
        "synthetic_ct_path": "",
        "reason": "",
        "total_sec": 0.0,
    }

    if not os.path.isdir(case_dir):
        result["status"] = "missing"
        result["reason"] = "No output directory"
        return result

    ct_path = os.path.join(case_dir, "synthetic_ct.nii.gz")
    result["has_synthetic_ct"] = os.path.isfile(ct_path)
    if result["has_synthetic_ct"]:
        result["synthetic_ct_path"] = ct_path

    result["has_nodmaisi_audit"] = os.path.isfile(
        os.path.join(case_dir, "nodmaisi_audit.json")
    )
    result["has_dataset_json"] = os.path.isfile(
        os.path.join(case_dir, "dataset.json")
    )
    result["has_input_mask"] = os.path.isfile(
        os.path.join(case_dir, "input_mask.nii.gz")
    )

    qc_slices = glob.glob(os.path.join(case_dir, "qc", "qc_slice_*.png"))
    result["has_qc_slices"] = len(qc_slices) > 0
    result["qc_slice_count"] = len(qc_slices)

    fail_path = os.path.join(case_dir, "nodmaisi_failure.json")
    result["has_failure_json"] = os.path.isfile(fail_path)

    # Read failure reason
    if result["has_failure_json"]:
        try:
            fdata = json.load(open(fail_path))
            result["reason"] = fdata.get("error", fdata.get("reason", ""))
        except Exception:
            result["reason"] = "unreadable failure json"

    # Read audit for timing / extra info
    audit_path = os.path.join(case_dir, "nodmaisi_audit.json")
    if os.path.isfile(audit_path):
        try:
            adata = json.load(open(audit_path))
            result["total_sec"] = adata.get("elapsed_sec", 0.0)
            if not result["reason"] and adata.get("status") == "failed":
                result["reason"] = adata.get("reason", "")
        except Exception:
            pass

    # Read per-case pipeline_summary for timing if audit didn't have it
    ps_path = os.path.join(case_dir, "pipeline_summary.json")
    if os.path.isfile(ps_path) and result["total_sec"] == 0.0:
        try:
            ps = json.load(open(ps_path))
            cases = ps.get("cases", [])
            if cases:
                result["total_sec"] = cases[0].get("total_sec", 0.0)
                if not result["reason"]:
                    result["reason"] = cases[0].get("reason", "")
        except Exception:
            pass

    # Determine overall status
    if result["has_synthetic_ct"] and result["has_nodmaisi_audit"] and result["has_qc_slices"]:
        result["status"] = "success"
    elif result["has_failure_json"]:
        result["status"] = "failed"
    elif result["has_synthetic_ct"] and not result["has_qc_slices"]:
        result["status"] = "incomplete"
        if not result["reason"]:
            result["reason"] = "synthetic_ct exists but QC slices missing"
    elif result["has_nodmaisi_audit"] and not result["has_synthetic_ct"]:
        result["status"] = "failed"
        if not result["reason"]:
            result["reason"] = "audit exists but no synthetic_ct produced"
    else:
        result["status"] = "incomplete"
        if not result["reason"]:
            missing = []
            if not result["has_synthetic_ct"]:
                missing.append("synthetic_ct")
            if not result["has_nodmaisi_audit"]:
                missing.append("nodmaisi_audit")
            if not result["has_qc_slices"]:
                missing.append("qc_slices")
            result["reason"] = f"Missing: {', '.join(missing)}"

    return result


def main():
    parser = argparse.ArgumentParser(description="Audit NodMAISI mode output")
    parser.add_argument("out_dir", help="Mode output directory (generated_cts/modeN_*)")
    parser.add_argument("--mask-dir", default="", help="Mask directory for expected case count")
    parser.add_argument("--output", default="", help="Output path for pipeline_summary.json (default: <out_dir>/pipeline_summary.json)")
    args = parser.parse_args()

    out_dir = args.out_dir
    if not os.path.isdir(out_dir):
        print(f"ERROR: Output directory not found: {out_dir}", file=sys.stderr)
        return 1

    mode_name = os.path.basename(out_dir)
    print(f"Auditing: {mode_name}")
    print(f"  Output dir: {out_dir}")

    # Discover case dirs
    case_dirs = discover_case_dirs(out_dir)
    print(f"  Case directories found: {len(case_dirs)}")

    # Discover expected cases from masks
    expected = discover_expected_cases(args.mask_dir)
    if expected:
        print(f"  Expected cases (from masks): {len(expected)}")

    # Identify cases that have no output dir at all
    missing_cases = sorted(expected - set(case_dirs.keys())) if expected else []

    # Audit each case dir
    results = []
    for cid, cdir in sorted(case_dirs.items()):
        results.append(audit_case(cid, cdir))

    # Add missing cases (no output dir)
    for cid in missing_cases:
        results.append({
            "case_id": cid,
            "status": "missing",
            "has_synthetic_ct": False,
            "has_nodmaisi_audit": False,
            "has_qc_slices": False,
            "qc_slice_count": 0,
            "has_failure_json": False,
            "has_dataset_json": False,
            "has_input_mask": False,
            "synthetic_ct_path": "",
            "reason": "No output directory",
            "total_sec": 0.0,
        })

    # Tally
    n_success = sum(1 for r in results if r["status"] == "success")
    n_failed = sum(1 for r in results if r["status"] == "failed")
    n_incomplete = sum(1 for r in results if r["status"] == "incomplete")
    n_missing = sum(1 for r in results if r["status"] == "missing")
    total = len(results)
    total_sec = sum(r["total_sec"] for r in results)

    # Failure reason breakdown
    failure_reasons = {}
    for r in results:
        if r["status"] in ("failed", "incomplete", "missing"):
            reason = r["reason"] or "unknown"
            failure_reasons.setdefault(reason, []).append(r["case_id"])

    # Print summary
    print(f"\n{'='*60}")
    print(f"  AUDIT SUMMARY: {mode_name}")
    print(f"{'='*60}")
    print(f"  Total cases:    {total}")
    if expected:
        print(f"  Expected:       {len(expected)}")
    print(f"  Success:        {n_success}  ({100*n_success/total:.1f}%)" if total else "")
    print(f"  Failed:         {n_failed}")
    print(f"  Incomplete:     {n_incomplete}")
    print(f"  Missing:        {n_missing}")
    print(f"  Total GPU time: {total_sec/3600:.1f} hours")

    if failure_reasons:
        print(f"\n  Failure/issue breakdown:")
        for reason, cases in sorted(failure_reasons.items(), key=lambda x: -len(x[1])):
            print(f"    [{len(cases):4d}] {reason[:100]}")

    # Build pipeline_summary.json
    summary = {
        "pipeline": "itrialspace_to_nodmaisi",
        "mode": mode_name,
        "timestamp": datetime.now().isoformat(),
        "audit_type": "post_run_aggregate",
        "total_cases": total,
        "expected_cases": len(expected) if expected else None,
        "n_success": n_success,
        "n_failed": n_failed,
        "n_incomplete": n_incomplete,
        "n_missing": n_missing,
        "success_rate": round(n_success / total, 4) if total else 0,
        "total_gpu_hours": round(total_sec / 3600, 2),
        "failure_breakdown": {
            reason: {
                "count": len(cases),
                "examples": cases[:5],
            }
            for reason, cases in sorted(
                failure_reasons.items(), key=lambda x: -len(x[1])
            )
        },
        "cases": results,
    }

    output_path = args.output or os.path.join(out_dir, "pipeline_summary.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Written: {output_path}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
