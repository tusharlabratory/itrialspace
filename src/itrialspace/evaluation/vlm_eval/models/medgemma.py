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
MedGemma 1.5 4B model adapter for generative VLM evaluation.

Uses the HuggingFace ``AutoModelForImageTextToText`` class with the
``google/medgemma-1.5-4b-it`` checkpoint (instruction-tuned MedGemma 1.5 4B).

MedGemma 1.5 is built on Gemma 3 with a medical-domain SigLIP vision encoder.
For CT imaging, MedGemma 1.5 expects a specific 3-channel RGB encoding where
each channel is a different HU window:
  R = wide (-1024, 1024)
  G = soft tissue (-135, 215)
  B = brain (0, 80)

The adapter supports multi-slice input: the centre slice + context slices
(e.g. _z-1, _z+1) are auto-discovered and passed as separate images in the
prompt, matching the official MedGemma CT notebook pattern.

Key characteristics:
- Built on Gemma 3 + SigLIP medical vision encoder
- Images resized to 896×896 and encoded to 256 tokens each by the processor
- Chat-style prompting with user role
- Multi-image support for multi-slice CT input
- ~4B parameters, ~8 GB VRAM in bfloat16
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import torch
from PIL import Image


class MedGemmaModel:
    """Generative VLM adapter for MedGemma 1.5 4B.

    Loads lazily on first call.  Produces free-text answers for a given
    image + text prompt which are then normalised by ``parsers.py``.
    """

    DEFAULT_MODEL_ID = "google/medgemma-1.5-4b-it"

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        max_new_tokens: int = 64,
    ):
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self._device_str = device
        self._cache_dir = cache_dir
        self._torch_dtype_str = torch_dtype
        self.max_new_tokens = max_new_tokens

        self._model = None
        self._processor = None
        self._device: Optional[torch.device] = None

    # ── lazy loading ─────────────────────────────────────────────────────

    def _load(self):
        if self._model is not None:
            return

        from transformers import AutoModelForImageTextToText, AutoProcessor

        if self._device_str:
            self._device = torch.device(self._device_str)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(
            self._torch_dtype_str or "",
            torch.bfloat16 if self._device.type == "cuda" else torch.float32,
        )

        print(f"Loading MedGemma from {self.model_id} ...")
        print(f"  Device: {self._device}  dtype: {torch_dtype}")

        # MedGemma is a GATED model: the access token is read from the environment
        # (.env -> HF_TOKEN), never hardcoded. See settings.hf_token / .env.example.
        from itrialspace.config.settings import hf_token

        token = hf_token()
        if token is None:
            print(
                "  WARN: no HF_TOKEN in environment/.env — MedGemma is gated and will "
                "fail to download unless you are already logged in (huggingface-cli login)."
            )

        kwargs: Dict = {"torch_dtype": torch_dtype}
        if self._cache_dir:
            kwargs["cache_dir"] = self._cache_dir
        if token:
            kwargs["token"] = token

        # `device_map="auto"` needs `accelerate`. Use it for multi-GPU sharding when
        # available; otherwise load normally and move to the device ourselves, so a
        # single-GPU run works without accelerate (and without crashing the job).
        try:
            from transformers.utils import is_accelerate_available

            use_device_map = is_accelerate_available()
        except Exception:
            use_device_map = False
        if use_device_map:
            kwargs["device_map"] = "auto"
        else:
            print("  accelerate not available — loading on a single device (no device_map).")

        # Try Flash Attention 2 for faster inference
        try:
            import flash_attn  # noqa: F401

            kwargs["attn_implementation"] = "flash_attention_2"
            print("  Using Flash Attention 2")
        except ImportError:
            pass

        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            use_fast=True,
            **({"cache_dir": self._cache_dir} if self._cache_dir else {}),
            **({"token": token} if token else {}),
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            **kwargs,
        )
        if not use_device_map:
            self._model = self._model.to(self._device)
        self._model.eval()
        print("  MedGemma loaded.")

    @property
    def device(self) -> torch.device:
        self._load()
        return self._device

    # ── image handling ───────────────────────────────────────────────────

    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        """Load an image as RGB.

        For png_medgemma_rgb slices the PNG is already 3-channel RGB
        (each channel = different HU window). For grayscale PNGs this
        converts to RGB by duplicating the single channel.
        """
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def discover_context_slices(center_path: str) -> List[str]:
        """Find the centre + context slice paths in sorted z-order.

        Given ``/dir/name.png`` looks for ``/dir/name_z-N.png`` and
        ``/dir/name_z+N.png`` siblings.  Returns the complete ordered
        list ``[..., _z-1, centre, _z+1, ...]``.
        """
        base, ext = os.path.splitext(center_path)
        pattern = re.compile(
            re.escape(os.path.basename(base)) + r"_z([+-]\d+)" + re.escape(ext) + "$"
        )
        parent = os.path.dirname(center_path) or "."
        neighbours: Dict[int, str] = {}
        for fname in os.listdir(parent):
            m = pattern.match(fname)
            if m:
                offset = int(m.group(1))
                neighbours[offset] = os.path.join(parent, fname)
        # Centre slice is offset 0
        neighbours[0] = center_path
        return [neighbours[k] for k in sorted(neighbours)]

    # ── inference ────────────────────────────────────────────────────────

    def generate(self, image_path: str, question: str) -> str:
        """Generate a free-text answer for CT slice(s) + question.

        Automatically discovers context slices (``_z-1``, ``_z+1``, etc.)
        stored alongside *image_path* and sends them all as separate
        images in the prompt, matching the MedGemma 1.5 CT protocol.

        Parameters
        ----------
        image_path : str
            Path to the **centre** slice PNG.
        question : str
            The instruction / question text.

        Returns
        -------
        str
            Raw generated text (decoded, prompt stripped).
        """
        self._load()

        # Collect ordered slice paths
        slice_paths = self.discover_context_slices(image_path)
        images = [self.load_image(p) for p in slice_paths]

        # Build multi-image chat prompt (official MedGemma CT pattern)
        content: List[Dict] = []
        for i, img in enumerate(images, 1):
            content.append({"type": "image", "image": img})
            if len(images) > 1:
                content.append({"type": "text", "text": f"SLICE {i}"})
        content.append({"type": "text", "text": question})

        messages = [{"role": "user", "content": content}]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device, dtype=torch.bfloat16)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Decode only the generated tokens (strip the input prompt)
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        answer = self._processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return answer
