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

"""Umbrella command-line interface for iTrialSpace (``its``).

Lightweight entry point over the core nodule index. Component-specific commands have
their own console scripts (``its-insert``, ``its-trials``, ``its-nodulemap``,
``its-retriever``).
"""

from __future__ import annotations

import argparse
import sys

from itrialspace import __version__
from itrialspace.config import settings


def _cmd_config(_args: argparse.Namespace) -> int:
    """Print resolved paths so users can verify their environment."""
    print("iTrialSpace resolved configuration:")
    print(f"  repo_root          {settings.repo_root()}")
    print(f"  data_dir           {settings.data_dir()}")
    print(f"  output_dir         {settings.output_dir()}")
    print(f"  configs_dir        {settings.configs_dir()}")
    print(f"  nodulemap_artifacts {settings.nodulemap_artifacts_dir()}")
    try:
        ds = settings.find_config("datasets.yaml")
        print(f"  datasets config    {ds}")
    except FileNotFoundError as exc:
        print(f"  datasets config    (not found: {exc})")
    return 0


def _load_index():
    from itrialspace import DatasetRegistry, NoduleIndex

    registry = DatasetRegistry.from_yaml()
    if len(registry) == 0:
        print(
            "No datasets found. Set ITRIALSPACE_DATA_DIR (see `its config`) and ensure "
            "profile CSVs exist; see docs/data.md.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return NoduleIndex.from_registry(registry)


def _cmd_stats(_args: argparse.Namespace) -> int:
    idx = _load_index()
    print(f"Loaded {len(idx)} nodules across {len(idx.datasets)} datasets.\n")
    print(idx.stats().to_string())
    return 0


def _cmd_info(_args: argparse.Namespace) -> int:
    from itrialspace import DatasetRegistry

    registry = DatasetRegistry.from_yaml()
    print(repr(registry))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="its", description="iTrialSpace command-line interface")
    parser.add_argument("--version", action="version", version=f"iTrialSpace {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_config = sub.add_parser("config", help="show resolved paths / environment")
    p_config.set_defaults(func=_cmd_config)

    p_info = sub.add_parser("info", help="list configured datasets")
    p_info.set_defaults(func=_cmd_info)

    p_index = sub.add_parser("index", help="nodule index operations")
    index_sub = p_index.add_subparsers(dest="index_command")
    p_stats = index_sub.add_parser("stats", help="build the index and print summary stats")
    p_stats.set_defaults(func=_cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return func(args)


if __name__ == "__main__":
    sys.exit(main())
