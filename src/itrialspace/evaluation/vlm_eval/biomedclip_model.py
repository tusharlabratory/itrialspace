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
BiomedCLIP model wrapper for image-text similarity scoring.

Uses the open_clip library to load
microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


class BiomedCLIPModel:
    """Thin wrapper around BiomedCLIP for image-text similarity tasks.

    The model is loaded lazily on first use. Supports both CPU and GPU.
    """

    MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    CONTEXT_LENGTH = 256

    def __init__(
        self,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self._device_str = device
        self._cache_dir = cache_dir
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = None

    def _load(self):
        """Lazy-load model, preprocess transform, and tokenizer via open_clip."""
        if self._model is not None:
            return

        from open_clip import create_model_from_pretrained, get_tokenizer

        if self._device_str:
            self._device = torch.device(self._device_str)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading BiomedCLIP from {self.MODEL_ID} ...")
        print(f"  Device: {self._device}")

        self._model, self._preprocess = create_model_from_pretrained(self.MODEL_ID)
        self._tokenizer = get_tokenizer(self.MODEL_ID)

        self._model.eval()
        self._model.to(self._device)
        print("  BiomedCLIP loaded.")

    @property
    def device(self) -> torch.device:
        self._load()
        return self._device

    def score_image_texts(
        self,
        image_path: str,
        texts: List[str],
    ) -> np.ndarray:
        """Compute cosine similarities between one image and multiple texts.

        Parameters
        ----------
        image_path : str
            Path to the PNG image.
        texts : list of str
            Text prompts to compare against.

        Returns
        -------
        np.ndarray
            Shape (len(texts),) of similarity scores (logits).
        """
        self._load()

        image = Image.open(image_path).convert("RGB")
        image_input = self._preprocess(image).unsqueeze(0).to(self._device)
        text_input = self._tokenizer(texts, context_length=self.CONTEXT_LENGTH).to(self._device)

        with torch.no_grad():
            image_features, text_features, logit_scale = self._model(image_input, text_input)
            similarities = logit_scale * image_features @ text_features.T

        return similarities.squeeze(0).cpu().numpy()

    def predict(
        self,
        image_path: str,
        labels: List[str],
        texts: List[str],
    ) -> Tuple[str, np.ndarray]:
        """Return the best-matching label and all similarity scores.

        Parameters
        ----------
        image_path : str
            Path to the PNG image.
        labels : list of str
            Label names corresponding to each text prompt.
        texts : list of str
            Text prompts.

        Returns
        -------
        (predicted_label, scores) : tuple
            The label with the highest score and the full score array.
        """
        scores = self.score_image_texts(image_path, texts)
        best_idx = int(np.argmax(scores))
        return labels[best_idx], scores

    def predict_batch(
        self,
        image_paths: List[str],
        labels: List[str],
        texts: List[str],
        batch_size: int = 64,
    ) -> List[Tuple[str, np.ndarray]]:
        """Batched prediction: process multiple images in one forward pass.

        Parameters
        ----------
        image_paths : list of str
            Paths to PNG images.
        labels : list of str
            Label names corresponding to each text prompt.
        texts : list of str
            Text prompts to compare against.
        batch_size : int
            Images per GPU forward pass.

        Returns
        -------
        list of (predicted_label, scores)
            One entry per image.
        """
        self._load()

        text_input = self._tokenizer(texts, context_length=self.CONTEXT_LENGTH).to(self._device)

        all_results: List[Tuple[str, np.ndarray]] = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = [self._preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            image_input = torch.stack(images).to(self._device)

            with torch.no_grad():
                image_features, text_features, logit_scale = self._model(image_input, text_input)
                similarities = logit_scale * image_features @ text_features.T

            scores_np = similarities.cpu().numpy()
            for j in range(scores_np.shape[0]):
                row_scores = scores_np[j]
                best_idx = int(np.argmax(row_scores))
                all_results.append((labels[best_idx], row_scores))

        return all_results
