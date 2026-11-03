"""
Rete neurale su mesh geodetica sferica (PyTorch).

- TriangleBlock: pesi locali + router ternario (3 lati)
- CausalAttnBlock: self-attention causale + FF (contesto temporale)
- SphericalMeshModel: emb → CausalAttn × N → walk mesh batched → LM head
- Pesi faccia impacchettati a coppie (pair packing) per buffer contigui GEMM-friendly
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mesh_topology import GeodesicMesh, build_geodesic_mesh, nearest_face


@dataclass
class WalkConfig:
    max_hops: int = 8
    # Se la probabilità del “exit” (soft) supera la soglia, si ferma prima
    exit_threshold: float = 0.55
    # Temperatura del router ternario
    router_temperature: float = 1.0
    # Se True, usa argmax (discreto); altrimenti Gumbel-Softmax straight-through
    hard_routing: bool = True


class TriangleBlock(nn.Module):
    """Micro-blocco su una faccia: MLP locale + logits verso 3 lati + exit."""

    def __init__(self, d_model: int, d_hidden: Optional[int] = None):
        super().__init__()
        h = d_hidden or d_model
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, h),
            nn.GELU(),
            nn.Linear(h, d_model),
        )
        # 3 lati + 1 “esci / emetti”
        self.router = nn.Linear(d_model, 4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (..., d)
        returns: (x_out, router_logits) con logits shape (..., 4)
        """
        h = self.ff(self.norm(x))
        x_out = x + h
        logits = self.router(x_out)
        return x_out, logits


class PairedFaceBank(nn.Module):
    """
    Banca pesi: per ogni coppia di facce adiacenti memorizza 2 TriangleBlock
    in un modulo ModuleList, con indici face→block per lookup O(1).
    I parametri di ogni coppia restano contigui nel ModuleList (ordine packing).
    """

    def __init__(self, mesh: GeodesicMesh, d_model: int, d_hidden: Optional[int] = None):
        super().__init__()
        self.num_faces = mesh.num_faces
        self.register_buffer("adjacency", torch.from_numpy(mesh.adjacency.astype(np.int64)))
        self.register_buffer("pairs", torch.from_numpy(mesh.pairs.astype(np.int64)))

        # face_id → index nel ModuleList `blocks`
        face_to_slot = np.full(mesh.num_faces, -1, dtype=np.int64)
        blocks: List[TriangleBlock] = []

        for fa, fb in mesh.pairs:
            face_to_slot[int(fa)] = len(blocks)
            blocks.append(TriangleBlock(d_model, d_hidden))
            face_to_slot[int(fb)] = len(blocks)
            blocks.append(TriangleBlock(d_model, d_hidden))

        for u in mesh.unpaired:
            face_to_slot[int(u)] = len(blocks)
            blocks.append(TriangleBlock(d_model, d_hidden))

        self.blocks = nn.ModuleList(blocks)
        self.register_buffer("face_to_slot", torch.from_numpy(face_to_slot))

    def forward_face(self, face_id: int, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        slot = int(self.face_to_slot[face_id].item())
        return self.blocks[slot](x)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention con maschera causale (lookahead vietato)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} non divisibile per n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = 1.0 / math.sqrt(self.head_dim)
        # PyTorch SDPA con is_causal=True (Flash/mem-efficient quando disponibili)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True,
            scale=scale,
        )  # (B, H, T, hd)
        y = y.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.resid_drop(self.proj(y))


class CausalAttnBlock(nn.Module):
    """Pre-norm: LayerNorm → causal attn → residual → LayerNorm → FF → residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        h = d_ff or (4 * d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, h),
            nn.GELU(),
            nn.Linear(h, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


def _default_n_heads(d_model: int) -> int:
    for h in (8, 4, 2, 1):
        if d_model % h == 0:
            return h
    return 1


class SphericalMeshModel(nn.Module):
    """
    LM: token → emb (+pos) → CausalAttn × N → walk mesh parallelo (B*T) → vocab.

    Contesto temporale via attenzione causale (no carry sequenziale).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        frequency: int = 2,
        walk: Optional[WalkConfig] = None,
        d_hidden: Optional[int] = None,
        n_attn_layers: int = 2,
        n_heads: Optional[int] = None,
        max_seq_len: int = 512,
        use_mesh_walk: bool = True,
        attn_dropout: float = 0.0,
        compass_spiral: bool = False,
    ):
        super().__init__()
        self.compass_spiral = bool(compass_spiral)
        self.mesh: GeodesicMesh = build_geodesic_mesh(
            frequency, compass_spiral=self.compass_spiral
        )
        self.walk = walk or WalkConfig()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_attn_layers = int(n_attn_layers)
        self.use_mesh_walk = bool(use_mesh_walk)
        self.max_seq_len = int(max_seq_len)
        heads = int(n_heads) if n_heads is not None else _default_n_heads(d_model)

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        self.pos_emb = nn.Embedding(self.max_seq_len, d_model)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        self.attn_blocks = nn.ModuleList(
            [
                CausalAttnBlock(d_model, heads, d_ff=d_hidden or (4 * d_model), dropout=attn_dropout)
                for _ in range(self.n_attn_layers)
            ]
        )

        # Proietta embedding → direzione 3D sulla sfera (faccia di ingresso)
        self.to_sphere = nn.Linear(d_model, 3)
        self.bank = PairedFaceBank(self.mesh, d_model, d_hidden)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # embedding tied (come goTommy2)
        self.lm_head.weight = self.tok_emb.weight
        self.emb_scale = d_model**-0.5

        # Buffer geometria per nearest face su GPU/CPU
        self.register_buffer(
            "centroids",
            torch.from_numpy(self.mesh.face_centroids.astype(np.float32)),
        )
        self.register_buffer(
            "adjacency",
            torch.from_numpy(self.mesh.adjacency.astype(np.int64)),
        )

    def project_to_face(self, h: torch.Tensor) -> torch.Tensor:
        """h (..., d) → face ids (...)"""
        direction = self.to_sphere(h)
        direction = F.normalize(direction, dim=-1)
        # (..., F)
        scores = direction @ self.centroids.T
        return scores.argmax(dim=-1)

    def _route_step(
        self,
        face: torch.Tensor,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        face: (B,) long — faccia corrente
        logits: (B, 4) — 3 lati + exit
        returns: next_face (B,), exit_mask (B,) bool, side_choice (B,) in {0,1,2,-1}
        """
        temp = max(self.walk.router_temperature, 1e-6)
        probs = F.softmax(logits / temp, dim=-1)
        exit_prob = probs[:, 3]
        side_probs = probs[:, :3]

        if self.walk.hard_routing:
            side = side_probs.argmax(dim=-1)
        else:
            # Gumbel-softmax straight-through sui 3 lati
            side = F.gumbel_softmax(logits[:, :3] / temp, hard=True, dim=-1).argmax(dim=-1)

        exit_mask = exit_prob >= self.walk.exit_threshold
        # lookup vettoriale: adjacency[face, side]
        next_face = self.adjacency[face.long(), side.long()]
        side_out = torch.where(exit_mask, torch.full_like(side, -1), side)
        return next_face, exit_mask, side_out

    def walk_token(
        self,
        h: torch.Tensor,
        start_face: Optional[torch.Tensor] = None,
        record_path: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[List[int]]]]:
        """
        h: (B, d) stato iniziale token — tipicamente B*T flattened
        returns: h_final (B, d), paths opzionale list per batch
        """
        b = h.shape[0]
        face = start_face if start_face is not None else self.project_to_face(h)
        paths: Optional[List[List[int]]] = [[int(face[i].item())] for i in range(b)] if record_path else None
        active = torch.ones(b, dtype=torch.bool, device=h.device)

        for _ in range(self.walk.max_hops):
            if not active.any():
                break
            # Esegue i blocchi per-faccia (batch misto: loop su facce uniche)
            logits = h.new_zeros(b, 4)
            h_new = h.clone()
            unique_faces = face[active].unique()
            for f_id in unique_faces.tolist():
                mask = active & (face == f_id)
                if not mask.any():
                    continue
                out, lg = self.bank.forward_face(int(f_id), h[mask])
                h_new[mask] = out
                logits[mask] = lg
            h = h_new

            next_face, exit_mask, _ = self._route_step(face, logits)
            # Solo i token ancora attivi possono uscire / muoversi
            moved = active & ~exit_mask
            face = torch.where(moved, next_face, face)
            just_exit = active & exit_mask
            active = active & ~exit_mask

            if record_path and paths is not None:
                for i in range(b):
                    if moved[i]:
                        paths[i].append(int(face[i].item()))
                    elif just_exit[i]:
                        paths[i].append(-1)  # sentinel exit

        return h, paths

    def forward(
        self,
        input_ids: torch.Tensor,
        record_path: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[List[List[int]]]]]:
        """
        input_ids: (B, T)
        returns: logits (B, T, V), paths [B][T][hops] se richiesto

        Pipeline:
          emb(+pos) → CausalAttnBlock × N → walk_token(B*T) → final_norm → lm_head
        """
        b, t = input_ids.shape
        if t > self.max_seq_len:
            raise ValueError(f"seq_len={t} > max_seq_len={self.max_seq_len}")

        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)  # (1, T)
        h = self.tok_emb(input_ids) * self.emb_scale + self.pos_emb(pos)

        for block in self.attn_blocks:
            h = block(h)  # (B, T, d)

        paths: Optional[List[List[List[int]]]] = None
        if self.use_mesh_walk:
            flat = h.reshape(b * t, self.d_model)
            flat_out, paths_flat = self.walk_token(flat, record_path=record_path)
            h = flat_out.view(b, t, self.d_model)
            if record_path and paths_flat is not None:
                # paths_flat[bi*T + ti] → paths[bi][ti]
                paths = [
                    [paths_flat[bi * t + ti] for ti in range(t)]
                    for bi in range(b)
                ]

        h = self.final_norm(h)
        logits = self.lm_head(h)
        # soft-cap: evita CE esplosa con residual walk + tied emb
        logits = 30.0 * torch.tanh(logits / 30.0)
        return logits, paths


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
