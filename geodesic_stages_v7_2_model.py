#!/usr/bin/env python3
"""Modello Geodetico a Sfere Piramidali Sentinel R-08 v7.2 (Sparse Upcycling / Warm-Start).
Architettura separata da v7 (non tocca né modifica alcun file o checkpoint di v7):
- Base: Geodesic Stages v7 (44.5M parametri, 3 stadi, frequenza 2, d=256, 2 layer attenzione).
- Modifica 1: Convoluzioni Causali 1D Inter-Stadio (k=4, d=256) tra Stadio 1->2 e 2->3 (Zero-Init Residual).
- Modifica 2: 12 Hub Regionali dell'Icosaedro per ciascuna piramide (partizione geometrica 80 facce -> 12 poli).
- Parametri Totali: 51.614.115 (~51.6M).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mesh_topology import build_geodesic_mesh, icosahedron_vertices, GeodesicMesh
from geodesic_stages_model import CascadeMeshModel, WalkConfig


def compute_face_to_icosahedron_regions(frequency: int = 2) -> torch.Tensor:
    """Partiziona geometricamente le facce della sfera geodetica verso il vertice dell'icosaedro più vicino."""
    mesh = build_geodesic_mesh(frequency)
    ico_v = icosahedron_vertices()  # (12, 3)
    centers = np.array([mesh.vertices[f].mean(axis=0) for f in mesh.faces])
    centers = centers / np.linalg.norm(centers, axis=-1, keepdims=True)
    sims = centers @ ico_v.T  # (n_faces, 12)
    regions = np.argmax(sims, axis=-1)
    return torch.from_numpy(regions).long()


class RegionalPyramidCenter(nn.Module):
    """Piramida Geodetica con 12 Centri Regionali indipendenti (1 per ciascun polo dell'Icosaedro).
    Risolve il collo di bottiglia dell'hub centrale singolo moltiplicando la capacità per 12.
    """
    def __init__(
        self,
        d_model: int = 256,
        n_faces: int = 80,
        d_narrow: int = 128,
        d_hidden: int = 512,
        n_regions: int = 12,
        face_to_region: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_faces = n_faces
        self.d_narrow = d_narrow
        self.d_hidden = d_hidden
        self.n_regions = n_regions

        if face_to_region is None:
            face_to_region = compute_face_to_icosahedron_regions(2)
        self.register_buffer("face_to_region", face_to_region.long().contiguous())

        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, d_narrow, bias=False)
        self.face_gain = nn.Parameter(torch.zeros(n_faces, d_narrow))
        self.face_shift = nn.Parameter(torch.zeros(n_faces, d_narrow))

        # 12 Centri MLP Regionali indipendenti
        self.centers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_narrow, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_model),
            )
            for _ in range(n_regions)
        ])
        self.reset_parameters(generator)

    def reset_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5), generator=generator)
        for c in self.centers:
            nn.init.kaiming_uniform_(c[0].weight, a=math.sqrt(5), generator=generator)
            if c[0].bias is not None:
                bound = 1.0 / math.sqrt(c[0].in_features)
                nn.init.uniform_(c[0].bias, -bound, bound, generator=generator)
            nn.init.zeros_(c[2].weight)
            nn.init.zeros_(c[2].bias)
        nn.init.zeros_(self.face_gain)
        nn.init.zeros_(self.face_shift)
        self.norm.reset_parameters()

    def forward(self, x: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model), face: (B, T)
        z = self.down(self.norm(x))
        z = z * (1.0 + F.embedding(face, self.face_gain)) + F.embedding(face, self.face_shift)

        reg = self.face_to_region[face]
        out = torch.empty_like(x)
        for r in range(self.n_regions):
            mask = (reg == r)
            if mask.any():
                out[mask] = self.centers[r](z[mask])
        return x + out


class CausalConv1D(nn.Module):
    """Convoluzione Causale 1D Inter-Stadio (Zero-Init Residual).
    Fornisce contesto temporale locale continuo token-to-token con zero KV-cache.
    Inizializzata rigorosamente a zero: a step 0 equivale all'identità perfetta.
    """
    def __init__(self, d_model: int = 256, kernel_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=0)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x_pad = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        out = self.conv(x_pad).transpose(1, 2)
        return x + out


class CascadeMeshModelV7_2(CascadeMeshModel):
    """Sentinel R-08 v7.2 con Convoluzioni Causali 1D e 12 Centri Regionali dell'Icosaedro."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        face_to_region = compute_face_to_icosahedron_regions(self.mesh.frequency)
        # Sostituzione delle piramidi con 12 Hub Regionali ciascuna
        self.pyramids = nn.ModuleList([
            RegionalPyramidCenter(
                d_model=self.d_model,
                n_faces=int(self.mesh.num_faces),
                d_narrow=self.pyramid_narrow,
                d_hidden=self.pyramid_hidden,
                n_regions=12,
                face_to_region=face_to_region,
            )
            for _ in self._banks()
        ])
        # Convoluzioni Causali 1D inter-stadio (Zero-Init Residual)
        self.conv_12 = CausalConv1D(d_model=self.d_model, kernel_size=4)
        self.conv_23 = CausalConv1D(d_model=self.d_model, kernel_size=4)

    def forward(
        self,
        pezzi_ids: torch.Tensor,
        parole_ids: Optional[torch.Tensor] = None,
        unioni_ids: Optional[torch.Tensor] = None,
        record_path: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[List[List[int]]]]]:
        b, t = pezzi_ids.shape
        if t > self.max_seq_len:
            raise ValueError(f"seq_len={t} > max_seq_len={self.max_seq_len}")

        pos = torch.arange(t, device=pezzi_ids.device).unsqueeze(0)
        h = self.emb_v1(pezzi_ids) * self.emb_scale + self.pos_emb(pos)

        for block in self.attn_blocks[:-1]:
            h = block(h)

        hashed = self.hash_routing
        h1, _ = self._walk_seq(
            h, self.bank1, self.to_sphere1, record_path=False, entry_ids=pezzi_ids if hashed else None
        )
        h1 = self._descend(0, h1, pezzi_ids)

        # Convoluzione Causale 1D tra Sfera 1 e Sfera 2
        h1_c = self.conv_12(h1)

        # Stage 2: fuse bilanciato + emb parole
        active2 = self._active_mask(1, parole_ids)
        emb2 = self.fuse_ln_e2(self.emb_v2(parole_ids))
        if self.fuse_mask_pad and active2 is not None:
            emb2 = emb2 * active2.unsqueeze(-1).to(dtype=emb2.dtype)
        h = self.fuse_norm2(self.fuse_ln_h2(self.proj_12(h1_c)) + emb2)
        if self.attn_inter2 is not None:
            h = self.attn_inter2(h)

        h2, _ = self._walk_seq(
            h,
            self.bank2,
            self.to_sphere2,
            record_path=False,
            entry_ids=parole_ids if hashed else None,
            active_bt=active2,
        )
        h2 = self._descend(1, h2, parole_ids, active_bt=active2)

        # Convoluzione Causale 1D tra Sfera 2 e Sfera 3
        h2_c = self.conv_23(h2)

        # Stage 3: fuse bilanciato + emb unioni
        active3 = self._active_mask(2, unioni_ids)
        emb3 = self.fuse_ln_e3(self.emb_v3(unioni_ids))
        if self.fuse_mask_pad and active3 is not None:
            emb3 = emb3 * active3.unsqueeze(-1).to(dtype=emb3.dtype)
        h = self.fuse_norm3(self.fuse_ln_h3(self.proj_23(h2_c)) + emb3)

        h3, paths = self._walk_seq(
            h,
            self.bank3,
            self.to_sphere3,
            record_path=record_path,
            entry_ids=unioni_ids if hashed else None,
            active_bt=active3,
        )
        h3 = self._descend(2, h3, unioni_ids, active_bt=active3)

        # Residual inter-stadio bilanciato + attn post-fuse
        h = (
            self.res_drop_h1(self.res_ln_h1(h1))
            + self.res_ln_h2(h2)
            + self.res_ln_h3(h3)
        )
        if len(self.attn_blocks) > 0:
            h = self.attn_blocks[-1](h)
        h = self.final_norm(h)
        logits = self.lm_head(h)
        logits = 30.0 * torch.tanh(logits / 30.0)
        return logits, paths
