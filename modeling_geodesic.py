# ============================================================================
# LEGAL NOTICE & WATERMARK: R08 NON-COMMERCIAL RESEARCH & ETHICAL LICENSE v2.0
# Copyright (c) 2026 Thomas Seri (Sentinel). All Rights Reserved.
# Exclusive Jurisdiction: Court of Milan (Sezione Specializzata Impresa), Italy.
# Clean-room reverse engineering, distillation, commercial or military use
# is strictly prohibited and constitutes a breach of contract.
# ============================================================================
__author__ = "Thomas Seri (Sentinel)"
__license__ = "R08-Non-Commercial-Ethical-License-v2.0"
__jurisdiction__ = "Court of Milan, Italy"

#!/usr/bin/env python3
"""R08-v7.2 (Rob) — Sentinel LifeOS Geodesic Mesh Neural Network (Pyramidal Sphere with 12 Regional Hubs & 1D Causal Convolutions, 51.6M)
Official Hugging Face / PyTorch implementation.
Author: Thomas Seri (https://www.sentinelhub.eu)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file

from geodesic_network import WalkConfig
from geodesic_stages_v7_2_model import CascadeMeshModelV7_2
import pezzi_tokenizer as pt
from vocab_stages import (
    StageBundle,
    StageEncoded,
    align_stages_to_pezzi,
    load_bundle,
    pass_to_v2,
    pass_to_v3,
)


def realign_from_pezzi(
    pezzi_ids: list[int], bundle: StageBundle, causal_align: bool = True
) -> Tuple[list[int], list[int], list[int]]:
    """Ricostruisce lo stato multi-stadio (pezzi, parole, unioni) in modo causale."""
    v2 = pass_to_v2(pezzi_ids, bundle)
    v3 = pass_to_v3(v2, bundle)
    enc = StageEncoded(
        pezzi_ids=list(pezzi_ids),
        parole_ids=list(v2.parole_ids),
        unioni_ids=list(v3.unioni_ids),
        spans=[],
        v2=v2,
        v3=v3,
    )
    p, w, u = align_stages_to_pezzi(enc, bundle, causal=causal_align)
    return list(p), list(w), list(u)


class R08Tokenizer:
    """Tokenizer a cascata a 3 stadi per R08-v7.2 (Rob)."""

    def __init__(self, bundle: StageBundle):
        self.bundle = bundle
        self.pezzi = bundle.pezzi
        self.eos_id = int(getattr(bundle.pezzi, "eos_id", 2))
        self.pad_id = int(getattr(bundle.pezzi, "pad_id", 3))

    @classmethod
    def from_pretrained(cls, pretrained_path: Union[str, Path]) -> "R08Tokenizer":
        root = Path(pretrained_path)
        vocab_dir = root / "vocab_stages" if (root / "vocab_stages").exists() else root
        bundle = load_bundle(vocab_dir)
        return cls(bundle)

    def encode(self, text: str) -> dict:
        """Codifica il testo nei 3 stadi sincronizzati (pezzi, parole, unioni)."""
        enc = pt.encode_cascade(text, self.bundle)
        p, w, u = align_stages_to_pezzi(enc, self.bundle, causal=True)
        return {
            "pezzi": list(p),
            "parole": list(w),
            "unioni": list(u),
        }

    def decode(self, token_ids: List[int]) -> str:
        """Decodifica gli ID token dei pezzi in testo UTF-8."""
        return pt.decode(token_ids, self.pezzi)


class R08Model(nn.Module):
    """Modello neurale geodetico-piramidale R08-v7.2 (Rob) da 51.6M parametri."""

    def __init__(self, config: dict, mesh_model: CascadeMeshModelV7_2):
        super().__init__()
        self.config = config
        self.mesh_model = mesh_model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: Union[str, Path],
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "R08Model":
        root = Path(pretrained_path)
        cfg_file = root / "config.json"
        if not cfg_file.exists():
            raise FileNotFoundError(f"config.json non trovato in {root}")

        with open(cfg_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        walk = WalkConfig(
            max_hops=1,
            exit_threshold=0.85,
            hard_routing=False,
            router_temperature=1.0,
        )

        mesh_model = CascadeMeshModelV7_2(
            vocab_pezzi=config.get("vocab_pezzi", 1251),
            vocab_parole=config.get("vocab_parole", 17908),
            vocab_unioni=config.get("vocab_unioni", 20006),
            d_model=config.get("d_model", 256),
            frequency=config.get("mesh_topology", {}).get("frequency", 2),
            walk=walk,
            n_attn_layers=config.get("n_attn_layers", 2),
            n_heads=config.get("n_heads", 8),
            max_seq_len=config.get("max_seq_len", 512),
            compass_spiral=config.get("routing", {}).get("compass_spiral", True),
            fast_walk=True,
            router_v6=config.get("routing", {}).get("router_v6", True),
            hash_routing=config.get("routing", {}).get("hash_routing", True),
            no_walk=config.get("routing", {}).get("no_walk", True),
            pyramid_center=config.get("pyramid_center", True),
        )

        weights_file = root / "model.safetensors"
        if not weights_file.exists():
            try:
                from huggingface_hub import hf_hub_download
                print("Scaricamento automatico di model.safetensors da Hugging Face (Tom2701/R-08-v7.2a)...")
                downloaded = hf_hub_download(
                    repo_id="Tom2701/R-08-v7.2a",
                    filename="model.safetensors",
                    local_dir=str(root),
                )
                weights_file = Path(downloaded)
            except Exception as e:
                raise FileNotFoundError(
                    f"model.safetensors non trovato in {root} e download automatico fallito ({e}). "
                    "Scaricalo da https://huggingface.co/Tom2701/R-08-v7.2a o dalle Release GitHub."
                )

        state_dict = load_file(str(weights_file))
        mesh_model.load_state_dict(state_dict, strict=True)

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        mesh_model.to(torch.device(device))
        if torch_dtype is not None:
            mesh_model.to(dtype=torch_dtype)

        mesh_model.eval()
        return cls(config, mesh_model)

    def forward(self, pezzi, parole, unioni, record_path: bool = False):
        return self.mesh_model(pezzi, parole, unioni, record_path=record_path)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        tokenizer: R08Tokenizer,
        max_new_tokens: int = 48,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
    ) -> str:
        """Generazione di testo autoregressiva con sampling controllato e penalità di ripetizione."""
        dev = next(self.mesh_model.parameters()).device
        enc = tokenizer.encode(prompt)
        p, w, u = enc["pezzi"], enc["parole"], enc["unioni"]
        if not p:
            p, w, u = [tokenizer.eos_id], [tokenizer.bundle.parole.pad_id], [tokenizer.bundle.unioni.pad_id]

        new_tokens: list[int] = []
        for _ in range(max_new_tokens):
            seq_p = torch.tensor([p[-256:]], dtype=torch.long, device=dev)
            seq_w = torch.tensor([w[-256:]], dtype=torch.long, device=dev)
            seq_u = torch.tensor([u[-256:]], dtype=torch.long, device=dev)

            logits, _ = self.mesh_model(seq_p, seq_w, seq_u, record_path=False)
            next_logits = logits[0, -1]

            # Repetition penalty
            if repetition_penalty != 1.0 and new_tokens:
                for token_id in set(new_tokens):
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= repetition_penalty
                    else:
                        next_logits[token_id] *= repetition_penalty

            # Temperature & Top-p Sampling
            if temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0.0
                    probs = probs / probs.sum()
                nxt = int(torch.multinomial(probs, num_samples=1).item())
            else:
                nxt = int(next_logits.argmax().item())

            if nxt in (tokenizer.eos_id, tokenizer.pad_id):
                break

            new_tokens.append(nxt)
            p.append(nxt)
            p, w, u = realign_from_pezzi(p, tokenizer.bundle, causal_align=True)

        return tokenizer.decode(new_tokens)
