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

# -*- coding: utf-8 -*-
"""
build_embeddings.py — generate embeddings from nodule profile features.

Usage (CLI):
    itrialspace-nodulemap build --model UMAP_2D --feature-set reinsertion_core --outdir ./artifacts

Programmatic:
    from itrialspace.apps.nodulemap.embeddings.build_embeddings import EmbeddingBuilder
    builder = EmbeddingBuilder.from_defaults()
    builder.build("UMAP_2D", "reinsertion_core", outdir="./artifacts")
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

_CONFIG_DIR = Path(__file__).parent.parent / "configs"


def _load_config() -> dict:
    """Load features.yaml config."""
    cfg_path = _CONFIG_DIR / "features.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


class FeaturePreprocessor:
    """Preprocesses raw profile DataFrame into a numeric feature matrix."""

    def __init__(
        self,
        numeric_cols: list[str],
        categorical_cols: list[str],
        scaler_type: str = "standard",
        imputer_strategy: str = "none",
    ):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.scaler_type = scaler_type
        self.imputer_strategy = imputer_strategy
        self._pipeline: Optional[ColumnTransformer] = None
        self._feature_names: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit preprocessor and transform data. Returns (n_samples, n_features)."""
        # Build numeric sub-pipeline
        num_steps = []
        if self.imputer_strategy != "none":
            num_steps.append(("impute", SimpleImputer(strategy=self.imputer_strategy)))
        if self.scaler_type == "robust":
            num_steps.append(("scale", RobustScaler()))
        else:
            num_steps.append(("scale", StandardScaler()))
        num_pipe = Pipeline(num_steps) if num_steps else "passthrough"

        # Build categorical sub-pipeline
        cat_pipe = (
            OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None)
            if self.categorical_cols
            else "passthrough"
        )

        transformers = []
        if self.numeric_cols:
            transformers.append(("num", num_pipe, self.numeric_cols))
        if self.categorical_cols:
            transformers.append(("cat", cat_pipe, self.categorical_cols))

        self._pipeline = ColumnTransformer(transformers, remainder="drop")

        # Replace inf values with NaN for imputation
        sub = df[self.numeric_cols + self.categorical_cols].copy()
        for c in self.numeric_cols:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
            sub[c] = sub[c].replace([np.inf, -np.inf], np.nan)

        X = self._pipeline.fit_transform(sub)

        # Store feature names
        self._feature_names = self._get_feature_names()
        return X.astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data with fitted preprocessor."""
        if self._pipeline is None:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        sub = df[self.numeric_cols + self.categorical_cols].copy()
        for c in self.numeric_cols:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
            sub[c] = sub[c].replace([np.inf, -np.inf], np.nan)
        return self._pipeline.transform(sub).astype(np.float32)

    def _get_feature_names(self) -> list[str]:
        """Extract feature names after fit."""
        names = []
        if self._pipeline is None:
            return names
        for name, trans, cols in self._pipeline.transformers_:
            if name == "num":
                names.extend(self.numeric_cols)
            elif name == "cat":
                if hasattr(trans, "get_feature_names_out"):
                    names.extend(trans.get_feature_names_out(self.categorical_cols).tolist())
                else:
                    names.extend(self.categorical_cols)
        return names

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names


class EmbeddingBuilder:
    """Orchestrates embedding generation from NoduleIndex."""

    def __init__(self, df: pd.DataFrame, verbose: bool = True):
        self._df = df
        self._verbose = verbose
        self._config = _load_config()

    @classmethod
    def from_defaults(cls, verbose: bool = True) -> "EmbeddingBuilder":
        """Load NoduleIndex from default registry."""
        from itrialspace.index.nodule_index import NoduleIndex
        from itrialspace.io.registry import DatasetRegistry

        # Resolve the dataset registry portably via itrialspace.config.settings
        # (local configs/datasets.yaml → shipped datasets.example.yaml → package default).
        registry = DatasetRegistry.from_yaml()
        if verbose:
            print("Loading NoduleIndex …")
        idx = NoduleIndex.from_registry(registry, verbose=verbose)
        return cls(idx.df, verbose=verbose)

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] {msg}")

    def list_feature_sets(self) -> list[str]:
        return list(self._config.get("feature_sets", {}).keys())

    def list_models(self) -> list[str]:
        return list(self._config.get("models", {}).keys())

    def build(
        self,
        model_name: str,
        feature_set: str = "reinsertion_core",
        outdir: str = "./nodulemap_artifacts",
        random_state: int = 42,
    ) -> dict:
        """
        Build embeddings and save artifacts.

        Returns dict with keys: embeddings, positions_2d (if 2D), metadata_path, etc.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # ── Load feature set config ─────────────────────────────────────────
        fs_cfg = self._config["feature_sets"].get(feature_set)
        if fs_cfg is None:
            raise ValueError(
                f"Unknown feature_set '{feature_set}'. Available: {self.list_feature_sets()}"
            )

        model_cfg = self._config["models"].get(model_name)
        if model_cfg is None:
            raise ValueError(f"Unknown model '{model_name}'. Available: {self.list_models()}")

        self._log(f"Feature set: {feature_set} ({fs_cfg['description']})")
        self._log(f"Model: {model_name} ({model_cfg['algorithm']}, {model_cfg['n_components']}D)")

        # ── Preprocess ──────────────────────────────────────────────────────
        self._log("Preprocessing features …")
        preprocessor = FeaturePreprocessor(
            numeric_cols=fs_cfg["numeric"],
            categorical_cols=fs_cfg.get("categorical", []),
            scaler_type=fs_cfg.get("scaler", "standard"),
            imputer_strategy=fs_cfg.get("imputer", "none"),
        )
        X = preprocessor.fit_transform(self._df)
        self._log(f"Feature matrix: {X.shape}")

        # Handle remaining NaNs (safety net)
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.any():
            n_bad = nan_mask.sum()
            self._log(f"WARNING: {n_bad} rows have NaN after preprocessing — filling with 0")
            X = np.nan_to_num(X, nan=0.0)

        # ── Build embedding ─────────────────────────────────────────────────
        algo = model_cfg["algorithm"]
        n_comp = model_cfg["n_components"]
        self._log(f"Computing {algo.upper()} embedding …")
        t0 = time.time()

        if algo == "pca":
            from sklearn.decomposition import PCA

            actual_comp = min(n_comp, X.shape[1], X.shape[0])
            if actual_comp < n_comp:
                self._log(
                    f"NOTE: capping PCA n_components {n_comp} → {actual_comp} "
                    f"(input has {X.shape[1]} features)"
                )
            reducer = PCA(n_components=actual_comp, random_state=random_state)
            embeddings = reducer.fit_transform(X)

        elif algo == "umap":
            import umap

            reducer = umap.UMAP(
                n_components=n_comp,
                n_neighbors=model_cfg.get("n_neighbors", 30),
                min_dist=model_cfg.get("min_dist", 0.3),
                metric=model_cfg.get("metric", "euclidean"),
                random_state=random_state,
                verbose=self._verbose,
            )
            embeddings = reducer.fit_transform(X)

        elif algo == "tsne":
            import inspect

            from sklearn.manifold import TSNE

            tsne_params = inspect.signature(TSNE.__init__).parameters
            # sklearn >= 1.6 renamed n_iter → max_iter
            iter_key = "max_iter" if "max_iter" in tsne_params else "n_iter"
            tsne_kwargs = dict(
                n_components=n_comp,
                perplexity=model_cfg.get("perplexity", 30),
                learning_rate=model_cfg.get("learning_rate", "auto"),
                random_state=random_state,
                verbose=1 if self._verbose else 0,
            )
            tsne_kwargs[iter_key] = model_cfg.get("n_iter", 1000)
            reducer = TSNE(**tsne_kwargs)
            embeddings = reducer.fit_transform(X)

        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        elapsed = time.time() - t0
        self._log(f"Embedding shape: {embeddings.shape} ({elapsed:.1f}s)")

        # ── Save artifacts ──────────────────────────────────────────────────
        tag = f"{feature_set}__{model_name}"

        # 1) Embeddings matrix
        emb_path = outdir / f"embeddings_{tag}.npy"
        np.save(emb_path, embeddings.astype(np.float32))
        self._log(f"Saved embeddings → {emb_path}")

        # 1b) Standardized feature matrix (per feature_set, reduced-free) — this is the
        # faithful space for similarity SEARCH (see neighbors/search_space.py). It is the
        # same across all models of a feature_set, so it is written once per feature_set.
        feat_path = outdir / f"features_{feature_set}.npy"
        np.save(feat_path, X.astype(np.float32))
        self._log(f"Saved feature matrix → {feat_path} {X.shape}")

        # 2) Node metadata table (parquet)
        meta_cols = [
            "dataset",
            "annotation_id",
            "patient_id",
            "label",
            "nodule_mean_diam_mm",
            "reinsertion_lobe",
            "reinsertion_lung_side",
            "reinsertion_lung_zone",
            "reinsertion_pleural_dist_mm",
            "reinsertion_airway_dist_mm",
            "reinsertion_nodule_diam_mm",
            "reinsertion_lobe_cc_pct",
            "reinsertion_lobe_ml_pct",
            "reinsertion_lobe_ap_pct",
            "reinsertion_lung_cc_pct",
            "reinsertion_lung_ml_pct",
            "reinsertion_lung_ap_pct",
            "ct_path",
            "coordX",
            "coordY",
            "coordZ",
        ]
        # Add optional columns that may exist
        optional = [
            "n_nodules_in_patient",
            "lobe_name",
            "lung_side",
            "lung_zone",
            "central_peripheral",
            "pleural_distance_mm",
            "airway_distance_mm",
        ]
        for c in optional:
            if c in self._df.columns and c not in meta_cols:
                meta_cols.append(c)

        # Add label metadata if present
        for c in ["label_source", "population_type"]:
            if c in self._df.columns:
                meta_cols.append(c)

        available_cols = [c for c in meta_cols if c in self._df.columns]
        meta_df = self._df[available_cols].copy().reset_index(drop=True)

        # Generate unique node ID: {dataset}_{annotation_id}
        meta_df.insert(
            0, "node_id", meta_df["dataset"] + "_" + meta_df["annotation_id"].astype(str)
        )

        # Add size bucket
        bins = [0, 5, 10, 15, 20, 30, 1000]
        labels = ["<5mm", "5-10mm", "10-15mm", "15-20mm", "20-30mm", ">30mm"]
        diam_col = (
            "reinsertion_nodule_diam_mm"
            if "reinsertion_nodule_diam_mm" in meta_df.columns
            else "nodule_mean_diam_mm"
        )
        meta_df["size_bucket"] = pd.cut(
            meta_df[diam_col].fillna(0), bins=bins, labels=labels
        ).astype(str)

        # Add label text
        meta_df["label_text"] = (
            meta_df["label"].map({0: "benign", 1: "malignant"}).fillna("unlabelled")
        )

        meta_path = outdir / f"metadata_{tag}.parquet"
        meta_df.to_parquet(meta_path, index=False)
        self._log(f"Saved metadata → {meta_path} ({len(meta_df)} rows)")

        # 3) Preprocessor pickle
        prep_path = outdir / f"preprocessor_{tag}.pkl"
        with open(prep_path, "wb") as f:
            pickle.dump(preprocessor, f)
        self._log(f"Saved preprocessor → {prep_path}")

        # 4) Model config record
        info = {
            "model_name": model_name,
            "feature_set": feature_set,
            "algorithm": algo,
            "n_components": n_comp,
            "n_samples": len(meta_df),
            "n_input_features": X.shape[1],
            "elapsed_seconds": round(elapsed, 2),
            "feature_names": preprocessor.feature_names,
            "model_config": model_cfg,
            "feature_set_config": fs_cfg,
        }
        info_path = outdir / f"info_{tag}.yaml"
        with open(info_path, "w") as f:
            yaml.safe_dump(info, f, sort_keys=False)
        self._log(f"Saved info → {info_path}")

        return {
            "embeddings_path": str(emb_path),
            "metadata_path": str(meta_path),
            "preprocessor_path": str(prep_path),
            "info_path": str(info_path),
            "embeddings_shape": embeddings.shape,
            "n_nodes": len(meta_df),
        }

    def build_all_2d(
        self,
        feature_set: str = "reinsertion_core",
        outdir: str = "./nodulemap_artifacts",
    ) -> dict:
        """Build PCA_2D + UMAP_2D + TSNE_2D for a given feature set."""
        results = {}
        for model in ["PCA_2D", "UMAP_2D", "TSNE_2D"]:
            self._log(f"\n{'='*60}\nBuilding {model} …\n{'='*60}")
            results[model] = self.build(model, feature_set, outdir)
        return results
