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

"""Central, machine-portable settings and path resolution for iTrialSpace.

This module is the **single source of truth** for every filesystem path the project
needs. Nothing else in the codebase should hardcode an absolute path. All locations
are derived from environment variables (with sensible defaults), so the same code runs
unchanged on a laptop, a workstation, or an HPC cluster.

Environment variables (see ``.env.example``):

    ITRIALSPACE_DATA_DIR    base directory of the unified data layout
                            (default: ~/.itrialspace/data)
    ITRIALSPACE_ROOT        repo root (auto-detected from this file when editable)
    ITRIALSPACE_OUTPUT_DIR  where generated artifacts go (default: ITRIALSPACE_DATA_DIR)

YAML interpolation:

    Any string value in a config YAML may reference these variables with ``${VAR}``
    syntax, e.g. ``base_dir: ${ITRIALSPACE_DATA_DIR}``. Use :func:`load_yaml` (or
    :func:`expand`) so the placeholders resolve — falling back to the documented
    defaults even when the variable is not exported.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Optional

_ENV_DATA = "ITRIALSPACE_DATA_DIR"
_ENV_ROOT = "ITRIALSPACE_ROOT"
_ENV_OUTPUT = "ITRIALSPACE_OUTPUT_DIR"
_ENV_NODULEMAP_ARTIFACTS = "NODULEMAP_ARTIFACTS"

# HuggingFace token env var (and the legacy alias transformers/hub also honour).
_ENV_HF_TOKEN = "HF_TOKEN"
_ENV_HF_TOKEN_LEGACY = "HUGGING_FACE_HUB_TOKEN"


def _parse_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (real env wins)."""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key.isidentifier():
            continue
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)  # never override an already-set variable


def load_dotenv_once() -> Optional[Path]:
    """Load the project's ``.env`` into ``os.environ`` exactly once.

    Mirrors ``infra/slurm/env.sh``: variables already in the real environment win,
    so a per-invocation override (or the SLURM-exported value) is never clobbered.
    Looks at ``$CWD/.env`` first, then the repo root's ``.env``. Secrets such as
    ``HF_TOKEN`` therefore come from ``.env`` (gitignored) and never from code.
    """
    if getattr(load_dotenv_once, "_done", False):
        return getattr(load_dotenv_once, "_path", None)
    load_dotenv_once._done = True  # type: ignore[attr-defined]
    candidates = [Path.cwd() / ".env"]
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            candidates.append(parent / ".env")
            break
    for envf in candidates:
        if envf.is_file():
            _parse_env_file(envf)
            load_dotenv_once._path = envf  # type: ignore[attr-defined]
            return envf
    return None


def hf_token() -> Optional[str]:
    """HuggingFace access token from the environment (``.env`` → ``HF_TOKEN``).

    Returns ``None`` when unset (public models still load; gated models such as
    MedGemma need a token — set ``HF_TOKEN`` in ``.env`` or the environment).
    """
    load_dotenv_once()
    return os.environ.get(_ENV_HF_TOKEN) or os.environ.get(_ENV_HF_TOKEN_LEGACY)


# Load .env as early as possible so every os.environ read below sees it.
load_dotenv_once()


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Best-effort repo root.

    Prefers ``$ITRIALSPACE_ROOT``; otherwise walks up from this file looking for a
    repo marker (``pyproject.toml``). Falls back to the current working directory
    when installed as a non-editable wheel.
    """
    env = os.environ.get(_ENV_ROOT)
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


@lru_cache(maxsize=1)
def data_dir() -> Path:
    """Base directory of the unified data layout (raw_ct/, masks/, profiles/, ...)."""
    env = os.environ.get(_ENV_DATA)
    base = Path(env).expanduser() if env else Path.home() / ".itrialspace" / "data"
    return base


@lru_cache(maxsize=1)
def output_dir() -> Path:
    """Where generated artifacts (manifests, masks, CTs, reports) are written."""
    env = os.environ.get(_ENV_OUTPUT)
    return Path(env).expanduser() if env else data_dir()


def configs_dir() -> Path:
    """Top-level ``configs/`` directory holding user/example configs."""
    return repo_root() / "configs"


def nodulemap_artifacts_dir() -> Path:
    """Directory holding NoduleMap build artifacts (embeddings, edges, metadata).

    Set ``$NODULEMAP_ARTIFACTS`` to read pre-built artifacts (or to choose where new
    ones are written). Default: ``$ITRIALSPACE_OUTPUT_DIR/nodulemap_artifacts``.
    """
    env = os.environ.get(_ENV_NODULEMAP_ARTIFACTS)
    return Path(env).expanduser() if env else output_dir() / "nodulemap_artifacts"


def _defaults() -> dict[str, str]:
    """Mapping used for ``${VAR}`` expansion: resolved defaults + live environment."""
    base = {
        _ENV_DATA: str(data_dir()),
        _ENV_ROOT: str(repo_root()),
        _ENV_OUTPUT: str(output_dir()),
        _ENV_NODULEMAP_ARTIFACTS: str(nodulemap_artifacts_dir()),
    }
    base.update(os.environ)  # real environment wins
    return base


def expand(value: Any) -> Any:
    """Expand ``${VAR}`` placeholders and ``~`` in a string.

    Non-strings are returned unchanged. Unknown variables are left intact (via
    ``safe_substitute``) so misconfigurations are visible rather than silently empty.
    """
    if not isinstance(value, str):
        return value
    substituted = Template(value).safe_substitute(_defaults())
    return os.path.expanduser(substituted)


def expand_tree(obj: Any) -> Any:
    """Recursively expand ``${VAR}`` in all string leaves of a dict/list structure."""
    if isinstance(obj, dict):
        return {k: expand_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_tree(v) for v in obj]
    return expand(obj)


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and expand ``${VAR}`` placeholders throughout."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML required: pip install pyyaml") from exc
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return expand_tree(raw)


def find_config(name: str, package_default: Optional[str | Path] = None) -> Path:
    """Resolve a config file name to a concrete path.

    Resolution order:
      1. ``$ITRIALSPACE_<NAME>_CONFIG`` env override (NAME upper-cased, no extension)
      2. top-level ``configs/<name>`` (user copy)
      3. top-level ``configs/<stem>.example.<ext>`` (shipped example)
      4. ``package_default`` (config bundled inside the package), if provided

    Raises ``FileNotFoundError`` if nothing is found.
    """
    stem = Path(name).stem
    ext = Path(name).suffix or ".yaml"

    env_key = f"ITRIALSPACE_{stem.upper()}_CONFIG"
    if os.environ.get(env_key):
        return Path(os.environ[env_key]).expanduser()

    cfg = configs_dir()
    candidates = [cfg / name, cfg / f"{stem}.example{ext}"]
    for cand in candidates:
        if cand.is_file():
            return cand

    if package_default is not None and Path(package_default).is_file():
        return Path(package_default)

    raise FileNotFoundError(
        f"Config '{name}' not found. Looked for: "
        + ", ".join(str(c) for c in candidates)
        + (f", {package_default}" if package_default else "")
        + f". Set {env_key} or create configs/{name}."
    )


__all__ = [
    "repo_root",
    "data_dir",
    "output_dir",
    "configs_dir",
    "nodulemap_artifacts_dir",
    "expand",
    "expand_tree",
    "load_yaml",
    "find_config",
]
