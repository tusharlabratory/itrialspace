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
LLaVA-Med model adapter for generative VLM evaluation.

Uses the HuggingFace ``LlavaForConditionalGeneration`` class.
Default model: ``llava-hf/llava-1.5-7b-hf`` (LLaVA-1.5 7B, fully HF-native).
Override ``model_id`` with a local LLaVA-Med checkpoint when available.

Key differences from BiomedCLIP:
- Generative (autoregressive) rather than contrastive
- Takes a chat-formatted prompt with ``<image>`` token
- Processor resizes images to 336×336 and normalises via CLIPImageProcessor
- Output is free-form text that must be parsed into canonical labels
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from PIL import Image


class LLaVAMedModel:
    """Generative VLM adapter around HuggingFace LLaVA.

    Loads lazily on first call.  Produces free-text answers for a given
    image + text prompt which are then normalised by ``parsers.py``.
    """

    DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

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

        from transformers import AutoProcessor, LlavaForConditionalGeneration

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
            torch.float16 if self._device.type == "cuda" else torch.float32,
        )

        print(f"Loading LLaVA from {self.model_id} ...")
        print(f"  Device: {self._device}  dtype: {torch_dtype}")

        # Access token read from the environment (.env -> HF_TOKEN); some LLaVA-Med
        # checkpoints are gated. Never hardcoded — see settings.hf_token / .env.example.
        from itrialspace.config.settings import hf_token

        token = hf_token()

        kwargs: Dict = {"torch_dtype": torch_dtype}
        if self._cache_dir:
            kwargs["cache_dir"] = self._cache_dir
        if token:
            kwargs["token"] = token

        # Try Flash Attention 2 for faster inference
        try:
            import flash_attn  # noqa: F401

            kwargs["attn_implementation"] = "flash_attention_2"
            print("  Using Flash Attention 2")
        except ImportError:
            pass

        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            **({"cache_dir": self._cache_dir} if self._cache_dir else {}),
            **({"token": token} if token else {}),
        )
        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            **kwargs,
        )
        self._model.eval()
        self._model.to(self._device)
        print("  LLaVA loaded.")

    @property
    def device(self) -> torch.device:
        self._load()
        return self._device

    # ── prompt formatting ────────────────────────────────────────────────

    def _format_prompt(self, question: str) -> str:
        """Build a chat-template prompt with an image placeholder."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        return self._processor.apply_chat_template(conversation, add_generation_prompt=True)

    # ── image preprocessing ──────────────────────────────────────────────

    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        """Load an image as RGB (LLaVA expects RGB input)."""
        return Image.open(image_path).convert("RGB")

    # ── inference ────────────────────────────────────────────────────────

    def generate(self, image_path: str, question: str) -> str:
        """Generate a free-text answer for one image + question pair.

        Parameters
        ----------
        image_path : str
            Path to PNG slice.
        question : str
            The instruction / question text.

        Returns
        -------
        str
            Raw generated text (decoded, prompt stripped).
        """
        self._load()

        image = self.load_image(image_path)
        prompt = self._format_prompt(question)

        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Strip input tokens from output to get only generated text
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        answer = self._processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return answer
