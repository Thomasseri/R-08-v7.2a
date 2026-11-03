"""
Cascade a 3 stadi in SERIE (non stream paralleli).

Architettura (allineata alla timeline pezzi, grain più fine):

  pezzi_ids  → emb_v1 (+pos) → CausalAttn_1 → mesh_walk_1 (=h1)
             → LN(proj_12(h1)) + LN(emb_v2)[*] → fuse_norm2 → mesh_walk_2 (=h2)
             → LN(proj_23(h2)) + LN(emb_v3)[*] → fuse_norm3 → mesh_walk_3 (=h3)
             → Drop0.1(n1(h1))+n2(h2)+n3(h3) → CausalAttn_2 → final_norm → lm_head (tied weight + bias)

  [*] v8.1 ``fuse_mask_pad``: su pad/unk V2/V3 l'addendo emb è azzerato (× active);
      ``mask_pad_stages`` già saltava solo walk/piramide.

Le sequenze parole/unioni sono espanse sulla lunghezza pezzi (pad dove non
c'è parola/unione). Obiettivo LM: next-token sui pezzi (unità di decode).

Router v6 (``router_v6=True``, default spento = v5 invariato), per stadio:
  - ingresso: faccia = argmax(coseni + face_bias) in training e in eval;
    face_bias (buffer, niente gradiente) segue il Loss-Free Balancing
    (``update_face_bias`` dopo ogni step); peso straight-through
    w = p_f / sg(p_f) con p = softmax(coseni · exp(logit_scale)), logit_scale
    appresa (init ln 10, clamp ≤ ln 100);
  - ogni salto: x + w · ff(norm(x)), il peso non tocca mai il residuo;
  - porte: come v5 (Gumbel-ST in training, argmax in eval);
  - uscita: decisione dura a soglia nel forward; il peso del salto successivo
    è w_porta · p_stay / sg(p_stay), p_stay = 1 − probs[:, uscita], così il
    logit di uscita riceve gradiente dal contributo dell'esperto che segue.
``hash_routing=True``: come v6 ma faccia d'ingresso fissa per id del token
(pezzo / parola / unione nei tre stadi, ``hash_faces``), senza to_sphere,
logit_scale né face_bias.
``no_walk=True`` (solo v6/hash, ``max_hops=1``): controllo del cammino, ogni
token usa solo il blocco della faccia d'ingresso in ogni stadio; dopo quel
blocco escono tutti e le porte non vengono valutate (niente rumore Gumbel).
``walk_to_center=True`` (v10 Spoke-to-Hub, solo hash, ``H_max∈{2,3}``):
ingresso Zipf/hash invariato (periferia = ID); dopo ogni blocco il token
segue ``parent_face[f]`` verso un **anello polare hub** di ``k_core≈15–20``
facce (non una sola faccia). Gate (Argon):
hop0 residuo pieno (come v8.1, ``g=None``);
hop≥1 ``g = (c + (1−c)·σ(Linear(x))) / √H_max`` con ``c=center_mix``
(default 0.25, formula liscia). Bias gate init 0. Niente Gumbel;
pad/unk = 0 hop (mask-pad). Train==eval path.
La piramide (se attiva) resta sulla faccia d'ingresso Zipf.
``sector_mix=True`` (v11, implica ``walk_to_center``): stesso albero parent→hub,
ma hop≥1 non è più spoke sottile solo-parent: ``x ← x + g · mean_{f∈settore(entry)}(Δ_f)``
dove il settore è lo **spicchio azimutale** di ``K≈sector_frac·F`` facce
(default 1/4) intorno all'ingresso Zipf. Hop0 resta solo faccia entry a peso 1.0.
Niente layer attention, niente Gumbel, Zipf classico (no multi-slot).
``sheet_mix=True`` (v13 Sheet-Mix / v14 pezzi-only, implica ``sector_mix`` path): stesso hop0/hop≥1
e ``_apply_sector_mean``, ma il buffer ``sector_faces`` è ``(F, K_eff=12)`` =
3 fogli complementari (cerchi massimi a 0°/90°/±45°, esclusa l'orientazione
``spiral_rank % 4``) × ``m=4`` facce equidistanti per foglio (ponte polo-a-polo).
Iso-param rispetto a v11 (solo indici; weight sharing invariato).
**Storico (v13–v18, default):** hop≥1 = media Δ sui blocchi delle K_eff facce dei fogli
(cross-face). **``sheet_entry_only=True`` (v19):** hop≥1 applica solo il blocco della
faccia d'ingresso Zipf (niente media su altre facce); i fogli non toccano altri esperti.
Il cammino parent→hub resta per lo stop; piramide invariata sull'entry.
``pyramid_center=True`` (solo hash, «sfera a spicchi piramidali»): dopo il
cammino di ogni stadio il token scende nella piramide della sua faccia
d'ingresso, un passaggio d → d/2 con proiezione condivisa e guadagno/spostamento
per faccia, e arriva al centro, un MLP denso condiviso dallo stadio; l'uscita
del centro si somma al ramo della faccia (``PyramidCenter``).
"""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_walk import (
    _build_paths,
    fast_walk_token,
    fast_walk_token_v6,
    grouped_block_forward,
    stack_bank,
)
from geodesic_network import (
    CausalAttnBlock,
    PairedFaceBank,
    WalkConfig,
    _default_n_heads,
)
from mesh_topology import GeodesicMesh, build_geodesic_mesh, icosahedron_vertices

DEFAULT_K_CORE = 18  # anello polare hub @ F=180 (~10%); clamp a [15, 20] tipico
V10_HOPS_DEFAULT = 2  # H_max=2; CLI può alzare a 3
DEFAULT_CENTER_MIX = 0.25  # hop≥1: g=(c+(1-c)σ(Wx))/√H; hop0=pieno
DEFAULT_SECTOR_FRAC = 0.25  # v11: fetta azimutale ~1/4 di F (~45 @ F=180)
DEFAULT_EDGE_NB_TAU = 4.0  # v12: softmax(τ · z_polo) sui 3 lati
DEFAULT_SHEET_M = 4  # v13: facce per foglio complementare (K_eff = 3·m = 12)


def build_face_edge_ids(faces: np.ndarray) -> Tuple[np.ndarray, int]:
    """``face_edges[f, i]`` = id unico del lato i; ``E = 3F/2`` su mesh chiusa."""
    fcount = int(faces.shape[0])
    edge_to_id: dict = {}
    face_edges = np.zeros((fcount, 3), dtype=np.int64)
    next_id = 0
    for fi, (a, b, c) in enumerate(faces):
        for li, (u, v) in enumerate(((int(a), int(b)), (int(b), int(c)), (int(c), int(a)))):
            key = (u, v) if u < v else (v, u)
            eid = edge_to_id.get(key)
            if eid is None:
                eid = next_id
                edge_to_id[key] = eid
                next_id += 1
            face_edges[fi, li] = eid
    return face_edges, next_id


def build_edge_neighbor_weights(
    adjacency: np.ndarray,
    face_centroids: np.ndarray,
    tau: float = DEFAULT_EDGE_NB_TAU,
) -> np.ndarray:
    """Pesi (F, 3) sui vicini di lato: softmax(τ · z_polo[nb]), verso il Polo Nord."""
    z = face_centroids[:, 2].astype(np.float64)
    scores = z[adjacency] * float(tau)  # (F, 3)
    scores = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(scores)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def build_spoke_hub_parent(
    adjacency: np.ndarray,
    face_centroids: np.ndarray,
    k_core: int = DEFAULT_K_CORE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Albero Spoke→Hub: ``parent[f]`` verso l'anello polare di ``k_core`` facce.

    Hub = top-``k_core`` facce per proiezione sull'asse polare icosaedrico
    (vertice 0, stesso di ``compass_spiral``). BFS multi-sorgente: ogni faccia
    non-hub punta al vicino più vicino all'anello; sulle hub ``parent[h]=h``.

    Ritorna ``(parent_face (F,), is_hub (F,) bool, hub_ids (K,))``.
    """
    n = int(adjacency.shape[0])
    k = int(max(1, min(int(k_core), n)))
    pole = icosahedron_vertices()[0]
    u3 = pole / np.linalg.norm(pole)
    z = face_centroids.astype(np.float64) @ u3
    hub_ids = np.argsort(-z)[:k].astype(np.int64)
    is_hub = np.zeros(n, dtype=bool)
    is_hub[hub_ids] = True
    parent = np.arange(n, dtype=np.int64)
    dist = np.full(n, -1, dtype=np.int32)
    q: deque = deque()
    for h in hub_ids.tolist():
        dist[h] = 0
        parent[h] = h
        q.append(h)
    while q:
        f = q.popleft()
        for nb in adjacency[f].tolist():
            nb = int(nb)
            if dist[nb] < 0:
                dist[nb] = int(dist[f]) + 1
                parent[nb] = f
                q.append(nb)
    if int((dist < 0).sum()) > 0:
        raise RuntimeError("spoke-hub: grafo facce non connesso")
    return parent, is_hub, hub_ids


def build_azimuth_sectors(
    face_centroids: np.ndarray,
    sector_frac: float = DEFAULT_SECTOR_FRAC,
) -> np.ndarray:
    """Settore azimutale per faccia: spicchio verso il polo (asse icosaedrico 0).

    Per ogni faccia ``e``, le ``K = max(1, round(sector_frac · F))`` facce con
    minore distanza circolare in azimut intorno al polo (stesso vertice 0 di
    ``build_spoke_hub_parent`` / ``compass_spiral``). Con ``sector_frac=0.25`` e
    F=180 → K≈45 (~1/4 di sfera in longitudine, tutte le latitudini).

    Ritorna ``sector_faces (F, K)`` int64; ``sector_faces[e, 0] == e``.
    """
    if not (0.0 < float(sector_frac) <= 1.0):
        raise ValueError("sector_frac deve essere in (0, 1]")
    cents = face_centroids.astype(np.float64)
    n = int(cents.shape[0])
    k = int(max(1, min(n, round(float(sector_frac) * n))))
    pole = icosahedron_vertices()[0]
    u3 = pole / np.linalg.norm(pole)
    # base ortonormale nel piano equatoriale
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(ref @ u3)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = ref - (ref @ u3) * u3
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(u3, e1)
    xy = cents - np.outer(cents @ u3, u3)
    x = xy @ e1
    y = xy @ e2
    az = np.arctan2(y, x)  # (-π, π]; poli (xy≈0) → 0
    # distanza circolare |Δφ| ∈ [0, π]
    dphi = np.abs(az[:, None] - az[None, :])
    dphi = np.minimum(dphi, 2.0 * np.pi - dphi)
    # self sempre in slot 0 (al polo più facce condividono φ≈0 → argsort non basta)
    np.fill_diagonal(dphi, -1.0)
    order = np.argsort(dphi, axis=1, kind="mergesort")[:, :k].astype(np.int64)
    return order


def build_complementary_sheets(
    face_centroids: np.ndarray,
    spiral_rank: Optional[np.ndarray] = None,
    m_per_sheet: int = DEFAULT_SHEET_M,
) -> np.ndarray:
    """v13 Sheet-Mix: 3 fogli complementari × m facce su cerchi massimi → (F, 3m).

    Per ogni faccia ``f`` costruisce la base tangente ``(t_par, t_mer)`` rispetto
    al polo icosaedrico (vertice 0). Le 4 normali ai cerchi massimi a
    0° / 90° / +45° / −45° sono::

        N_V  = t_par          (meridiano, polo↔polo)
        N_H  = t_mer          (ortogonale in f)
        N_+  = (t_par+t_mer)/√2
        N_-  = (t_par-t_mer)/√2

    L'orientazione propria è ``spiral_rank[f] % 4`` (se assente → 0); si usano
    i **3 fogli complementari**. Su ciascun foglio si campionano ``m`` facce
    equidistanti sul cerchio massimo (θ = 2πk/m), incluso il ponte
    fronte/retro e polo-a-polo. Drop-in su ``sector_faces`` / ``_apply_sector_mean``.
    """
    m = int(m_per_sheet)
    if m < 1:
        raise ValueError("m_per_sheet deve essere >= 1")
    cents = face_centroids.astype(np.float64)
    n = int(cents.shape[0])
    # normalizza (centroidi mesh già unitari, ma per sicurezza)
    norms = np.linalg.norm(cents, axis=1, keepdims=True)
    cents = cents / np.maximum(norms, 1e-12)
    pole = icosahedron_vertices()[0]
    u3 = pole / np.linalg.norm(pole)
    # asse equatoriale fisso per i poli (stesso schema di build_azimuth_sectors)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(ref @ u3)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = ref - (ref @ u3) * u3
    e1 = e1 / np.linalg.norm(e1)

    if spiral_rank is None:
        orient = np.zeros(n, dtype=np.int64)
    else:
        sr = np.asarray(spiral_rank, dtype=np.int64).reshape(-1)
        if sr.shape[0] != n:
            raise ValueError(f"spiral_rank len={sr.shape[0]} ≠ F={n}")
        orient = np.mod(sr, 4).astype(np.int64)

    out = np.empty((n, 3 * m), dtype=np.int64)
    # precompute all pairwise cosines for nearest-face lookup
    # (F=180 → F² cheap)
    for f in range(n):
        c = cents[f]
        # t_par ∝ u3 × c  (parallelo); ai poli |u3×c|→0 → e1
        t_par = np.cross(u3, c)
        tp_n = float(np.linalg.norm(t_par))
        if tp_n < 1e-8:
            t_par = e1.copy()
        else:
            t_par = t_par / tp_n
        t_mer = np.cross(c, t_par)
        tm_n = float(np.linalg.norm(t_mer))
        if tm_n < 1e-8:
            t_mer = np.cross(t_par, u3)
            t_mer = t_mer / max(float(np.linalg.norm(t_mer)), 1e-12)
        else:
            t_mer = t_mer / tm_n
        normals = (
            t_par,  # 0 Verticale
            t_mer,  # 1 Orizzontale
            (t_par + t_mer) / math.sqrt(2.0),  # 2 D+
            (t_par - t_mer) / math.sqrt(2.0),  # 3 D-
        )
        own = int(orient[f])
        comps = [s for s in range(4) if s != own]
        col = 0
        for s in comps:
            N = normals[s]
            # base ortonormale nel piano del cerchio massimo (⊥ N), con ea ≈ c
            ea = c - (c @ N) * N
            ea_n = float(np.linalg.norm(ea))
            if ea_n < 1e-8:
                # c ∥ N (non dovrebbe succedere: N costruita ⊥ c)
                ea = t_par if s != 0 else t_mer
                ea = ea - (ea @ N) * N
                ea = ea / max(float(np.linalg.norm(ea)), 1e-12)
            else:
                ea = ea / ea_n
            eb = np.cross(N, ea)
            eb = eb / max(float(np.linalg.norm(eb)), 1e-12)
            for k in range(m):
                theta = 2.0 * math.pi * float(k) / float(m)
                target = math.cos(theta) * ea + math.sin(theta) * eb
                # faccia col centroide più allineato a target
                scores = cents @ target
                out[f, col] = int(np.argmax(scores))
                col += 1
        # slot 0 = entry (coerente con build_azimuth_sectors)
        out[f, 0] = f
    return out


V6_LOGIT_SCALE_INIT = math.log(10.0)
V6_LOGIT_SCALE_MAX = math.log(100.0)
V6_BIAS_RATE = 1e-3
_HASH_SALTS = (0x9E3779B9, 0x85EBCA6B, 0xC2B2AE35)
PYRAMID_SEED_OFFSET = 7919  # seme del generatore delle piramidi = seme torch + offset


def _lowbias32(x: int) -> int:
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def hash_faces(vocab_size: int, n_faces: int, stage: int) -> torch.Tensor:
    """Faccia d'ingresso fissa per id (--hash-routing): lowbias32(id XOR sale dello stadio) mod n_faces."""
    salt = _HASH_SALTS[stage - 1]
    return torch.tensor([_lowbias32(i ^ salt) % n_faces for i in range(vocab_size)], dtype=torch.long)


class PyramidCenter(nn.Module):
    """Piramidi sotto le facce di uno stadio e centro condiviso (--pyramid-center).

    Discesa dalla faccia f: z = (1 + face_gain[f]) ⊙ down(norm(x)) + face_shift[f], d → d_narrow;
    la proiezione propria della faccia è diag(1 + face_gain[f]) · down. Centro, uguale per tutti
    i token: c = up(GELU(inner(z))), d_narrow → d_hidden → d. Uscita x + c; up parte da zero, quindi
    all'inizio l'uscita è x.
    """

    def __init__(
        self, d_model: int, n_faces: int, d_narrow: int, d_hidden: int, generator: Optional[torch.Generator] = None
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, d_narrow, bias=False)
        self.face_gain = nn.Parameter(torch.zeros(n_faces, d_narrow))
        self.face_shift = nn.Parameter(torch.zeros(n_faces, d_narrow))
        self.center = nn.Sequential(nn.Linear(d_narrow, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_model))
        self.reset_parameters(generator)

    def reset_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        """Init di nn.Linear per down e per il primo strato del centro; ultima proiezione, guadagni e spostamenti a zero."""
        for lin in (self.down, self.center[0]):
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5), generator=generator)
            if lin.bias is not None:
                bound = 1.0 / math.sqrt(lin.in_features)
                nn.init.uniform_(lin.bias, -bound, bound, generator=generator)
        for p in (self.center[2].weight, self.center[2].bias, self.face_gain, self.face_shift):
            nn.init.zeros_(p)
        self.norm.reset_parameters()

    def forward(self, x: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        z = self.down(self.norm(x))
        z = z * (1.0 + F.embedding(face, self.face_gain)) + F.embedding(face, self.face_shift)
        return x + self.center(z)


class HexPyramidCenter(nn.Module):
    """v9: guadagno/spostamento per **vertice** della mesh; media sui 3 vertici della faccia.

    Stessa discesa d → d_narrow e centro MLP condiviso di ``PyramidCenter``, ma i parametri
    locali vivono sui V vertici (esagoni/pentagoni duali) invece che sulle F facce: facce
    adiacenti condividono i vertici e quindi parte della piramide.
    """

    def __init__(
        self,
        d_model: int,
        n_vertices: int,
        face_vertices: torch.Tensor,
        d_narrow: int,
        d_hidden: int,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        if face_vertices.ndim != 2 or face_vertices.shape[1] != 3:
            raise ValueError(f"face_vertices deve essere (F, 3), got {tuple(face_vertices.shape)}")
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, d_narrow, bias=False)
        self.vertex_gain = nn.Parameter(torch.zeros(n_vertices, d_narrow))
        self.vertex_shift = nn.Parameter(torch.zeros(n_vertices, d_narrow))
        self.center = nn.Sequential(nn.Linear(d_narrow, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_model))
        self.register_buffer("face_vertices", face_vertices.long().contiguous())
        self.reset_parameters(generator)

    def reset_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        for lin in (self.down, self.center[0]):
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5), generator=generator)
            if lin.bias is not None:
                bound = 1.0 / math.sqrt(lin.in_features)
                nn.init.uniform_(lin.bias, -bound, bound, generator=generator)
        for p in (self.center[2].weight, self.center[2].bias, self.vertex_gain, self.vertex_shift):
            nn.init.zeros_(p)
        self.norm.reset_parameters()

    def forward(self, x: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        z = self.down(self.norm(x))
        verts = self.face_vertices[face]  # (..., 3)
        g = F.embedding(verts, self.vertex_gain).mean(dim=-2)
        s = F.embedding(verts, self.vertex_shift).mean(dim=-2)
        z = z * (1.0 + g) + s
        return x + self.center(z)


class SharedEdgeVertexPyramid(nn.Module):
    """v12: face_gain privato + lati (E) e vertici (V) condivisi + 1 centro MLP.

    ``z`` modulato da ``1 + face + 0.5·mean(edge) + 0.5·mean(vertex)`` (edge/vert init 0).
    """

    def __init__(
        self,
        d_model: int,
        n_faces: int,
        n_edges: int,
        n_vertices: int,
        face_edges: torch.Tensor,
        face_vertices: torch.Tensor,
        d_narrow: int,
        d_hidden: int,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        if face_edges.ndim != 2 or face_edges.shape[1] != 3:
            raise ValueError(f"face_edges deve essere (F, 3), got {tuple(face_edges.shape)}")
        if face_vertices.ndim != 2 or face_vertices.shape[1] != 3:
            raise ValueError(f"face_vertices deve essere (F, 3), got {tuple(face_vertices.shape)}")
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, d_narrow, bias=False)
        self.face_gain = nn.Parameter(torch.zeros(n_faces, d_narrow))
        self.face_shift = nn.Parameter(torch.zeros(n_faces, d_narrow))
        self.edge_gain = nn.Parameter(torch.zeros(n_edges, d_narrow))
        self.edge_shift = nn.Parameter(torch.zeros(n_edges, d_narrow))
        self.vertex_gain = nn.Parameter(torch.zeros(n_vertices, d_narrow))
        self.vertex_shift = nn.Parameter(torch.zeros(n_vertices, d_narrow))
        self.center = nn.Sequential(nn.Linear(d_narrow, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_model))
        self.register_buffer("face_edges", face_edges.long().contiguous())
        self.register_buffer("face_vertices", face_vertices.long().contiguous())
        self.reset_parameters(generator)

    def reset_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        for lin in (self.down, self.center[0]):
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5), generator=generator)
            if lin.bias is not None:
                bound = 1.0 / math.sqrt(lin.in_features)
                nn.init.uniform_(lin.bias, -bound, bound, generator=generator)
        for p in (
            self.center[2].weight,
            self.center[2].bias,
            self.face_gain,
            self.face_shift,
            self.edge_gain,
            self.edge_shift,
            self.vertex_gain,
            self.vertex_shift,
        ):
            nn.init.zeros_(p)
        self.norm.reset_parameters()

    def forward(self, x: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        z = self.down(self.norm(x))
        f = face.long()
        eg = F.embedding(self.face_edges[f], self.edge_gain).mean(dim=-2)
        es = F.embedding(self.face_edges[f], self.edge_shift).mean(dim=-2)
        vg = F.embedding(self.face_vertices[f], self.vertex_gain).mean(dim=-2)
        vs = F.embedding(self.face_vertices[f], self.vertex_shift).mean(dim=-2)
        g = F.embedding(f, self.face_gain) + 0.5 * eg + 0.5 * vg
        s = F.embedding(f, self.face_shift) + 0.5 * es + 0.5 * vs
        z = z * (1.0 + g) + s
        return x + self.center(z)


def pyramid_kwargs(blob: dict) -> dict:
    """Argomenti di CascadeMeshModel per la piramide, dai metadati di un checkpoint (assenti = spenta)."""
    if not blob.get("pyramid_center") and not blob.get("hex_pyramid") and not blob.get("shared_edges_verts"):
        return {}
    cfg = blob.get("pyramid_config") or {}
    out: dict = {
        "pyramid_center": True,
        "pyramid_narrow": cfg.get("d_stretto"),
        "pyramid_hidden": cfg.get("d_centro"),
    }
    if blob.get("hex_pyramid"):
        out["hex_pyramid"] = True
    if blob.get("shared_edges_verts"):
        out["shared_edges_verts"] = True
    return out


def v8_kwargs(blob: dict) -> dict:
    """Argomenti v8.x di CascadeMeshModel dai metadati checkpoint (assenti = default v7).

    Lo zipf-hash è già nei buffer ``bank.hash_face`` dello state_dict: non serve
    ``zipf_hash_path`` in eval.

    ``fuse_mask_pad`` (v8.1): solo se presente nel blob — i ckpt v8.0 senza chiave
    restano senza mask sull'addendo emb in fuse (come in training).
    """
    out: dict = {}
    if blob.get("d_hidden") is not None:
        out["d_hidden"] = int(blob["d_hidden"])
    de = blob.get("d_emb_v23")
    if de is not None and int(de) > 0:
        out["d_emb_v23"] = int(de)
    if blob.get("inter_stage_attn"):
        out["inter_stage_attn"] = True
    if blob.get("mask_pad_stages"):
        out["mask_pad_stages"] = True
        out["pad_id_v2"] = int(blob.get("pad_id_v2", 3))
        out["unk_id_v2"] = int(blob.get("unk_id_v2", 4))
        out["pad_id_v3"] = int(blob.get("pad_id_v3", 3))
        out["unk_id_v3"] = int(blob.get("unk_id_v3", 4))
    if blob.get("fuse_mask_pad"):
        out["fuse_mask_pad"] = True
    if blob.get("multi_slot_v1"):
        out["multi_slot_v1"] = True
    return out


def _make_emb(vocab: int, d_emb: int, d_model: int) -> nn.Module:
    """Embedding pieno a ``d_model``, oppure ``Embedding(d_emb)→Linear→d_model``."""
    if d_emb >= d_model:
        emb = nn.Embedding(vocab, d_model)
        nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        return emb
    emb = nn.Embedding(vocab, d_emb)
    proj = nn.Linear(d_emb, d_model, bias=False)
    nn.init.normal_(emb.weight, mean=0.0, std=0.02)
    nn.init.normal_(proj.weight, mean=0.0, std=0.02)
    return nn.Sequential(emb, proj)


class CascadeMeshModel(nn.Module):
    """Tre emb + tre mesh in catena (default), oppure solo-V1 (v1_only=True)."""

    def __init__(
        self,
        vocab_pezzi: int,
        vocab_parole: int = 0,
        vocab_unioni: int = 0,
        d_model: int = 256,
        frequency: int = 2,
        walk: Optional[WalkConfig] = None,
        d_hidden: Optional[int] = None,
        n_attn_layers: int = 2,
        n_heads: Optional[int] = None,
        max_seq_len: int = 512,
        attn_dropout: float = 0.0,
        res_h1_dropout: float = 0.1,
        v1_only: bool = False,
        compass_spiral: bool = False,
        fast_walk: bool = False,
        router_v6: bool = False,
        hash_routing: bool = False,
        no_walk: bool = False,
        walk_to_center: bool = False,
        k_core: int = DEFAULT_K_CORE,
        center_mix: float = DEFAULT_CENTER_MIX,
        sector_mix: bool = False,
        sector_frac: float = DEFAULT_SECTOR_FRAC,
        sheet_mix: bool = False,
        sheet_m: int = DEFAULT_SHEET_M,
        sheet_entry_only: bool = False,
        edge_neighbor_mix: bool = False,
        edge_nb_tau: float = DEFAULT_EDGE_NB_TAU,
        pyramid_center: bool = False,
        pyramid_narrow: Optional[int] = None,
        pyramid_hidden: Optional[int] = None,
        hex_pyramid: bool = False,
        shared_edges_verts: bool = False,
        multi_slot_v1: bool = False,
        d_emb_v23: Optional[int] = None,
        inter_stage_attn: bool = False,
        mask_pad_stages: bool = False,
        fuse_mask_pad: bool = False,
        zipf_hash_path: Optional[Union[str, Path]] = None,
        pad_id_v2: int = 3,
        unk_id_v2: int = 4,
        pad_id_v3: int = 3,
        unk_id_v3: int = 4,
    ):
        super().__init__()
        self.compass_spiral = bool(compass_spiral)
        # walk_token vettorizzato (fast_walk.py): stessi risultati del loop per faccia, senza sync per faccia
        self.fast_walk = bool(fast_walk)
        self.mesh: GeodesicMesh = build_geodesic_mesh(
            frequency, compass_spiral=self.compass_spiral
        )
        self.walk = walk or WalkConfig()
        self.d_model = int(d_model)
        self.vocab_pezzi = int(vocab_pezzi)
        self.vocab_parole = int(vocab_parole)
        self.vocab_unioni = int(vocab_unioni)
        self.v1_only = bool(v1_only)
        self.n_attn_layers = int(n_attn_layers)
        self.max_seq_len = int(max_seq_len)
        self.d_hidden = int(d_hidden) if d_hidden is not None else None
        self.d_emb_v23 = int(d_emb_v23) if d_emb_v23 is not None else None
        self.inter_stage_attn = bool(inter_stage_attn)
        self.mask_pad_stages = bool(mask_pad_stages)
        # v8.1: azzera anche l'addendo emb in fuse su pad/unk (0 nuovi param).
        # Richiede mask_pad_stages; se True senza mask_pad viene ignorato in forward.
        self.fuse_mask_pad = bool(fuse_mask_pad) and bool(mask_pad_stages)
        self.pad_id_v2 = int(pad_id_v2)
        self.unk_id_v2 = int(unk_id_v2)
        self.pad_id_v3 = int(pad_id_v3)
        self.unk_id_v3 = int(unk_id_v3)
        heads = int(n_heads) if n_heads is not None else _default_n_heads(d_model)
        # attn backbone sempre 4·d; d_hidden slimma solo le FFN delle facce (v8)
        attn_ff = 4 * d_model

        self.emb_v1 = nn.Embedding(vocab_pezzi, d_model)
        nn.init.normal_(self.emb_v1.weight, mean=0.0, std=0.02)
        if not self.v1_only:
            if vocab_parole <= 0 or vocab_unioni <= 0:
                raise ValueError(
                    "Cascade 3-stadi richiede vocab_parole > 0 e vocab_unioni > 0 "
                    "(oppure v1_only=True)"
                )
            de = self.d_emb_v23 if self.d_emb_v23 is not None else d_model
            self.emb_v2 = _make_emb(vocab_parole, de, d_model)
            self.emb_v3 = _make_emb(vocab_unioni, de, d_model)

        self.pos_emb = nn.Embedding(self.max_seq_len, d_model)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        self.emb_scale = d_model**-0.5

        self.attn_blocks = nn.ModuleList(
            [
                CausalAttnBlock(d_model, heads, d_ff=attn_ff, dropout=attn_dropout)
                for _ in range(self.n_attn_layers)
            ]
        )
        self.attn_inter2: Optional[CausalAttnBlock] = None
        if self.inter_stage_attn and not self.v1_only:
            self.attn_inter2 = CausalAttnBlock(d_model, heads, d_ff=attn_ff, dropout=attn_dropout)

        # Banca mesh V1 (80 triangoli a freq=2); stadi 2/3 creati solo se non v1_only
        self.to_sphere1 = nn.Linear(d_model, 3)
        self.bank1 = PairedFaceBank(self.mesh, d_model, d_hidden)
        if not self.v1_only:
            self.to_sphere2 = nn.Linear(d_model, 3)
            self.to_sphere3 = nn.Linear(d_model, 3)
            self.bank2 = PairedFaceBank(self.mesh, d_model, d_hidden)
            self.bank3 = PairedFaceBank(self.mesh, d_model, d_hidden)

            # Fuse bilanciato: LN per-addendo (niente emb_scale su V2/V3 — spegneva gli emb)
            self.proj_12 = nn.Linear(d_model, d_model)
            self.proj_23 = nn.Linear(d_model, d_model)
            self.fuse_ln_h2 = nn.LayerNorm(d_model)
            self.fuse_ln_e2 = nn.LayerNorm(d_model)
            self.fuse_ln_h3 = nn.LayerNorm(d_model)
            self.fuse_ln_e3 = nn.LayerNorm(d_model)
            self.fuse_norm2 = nn.LayerNorm(d_model)
            self.fuse_norm3 = nn.LayerNorm(d_model)

        # Residual bilanciato: LN per-stadio così ||h1||~180 non schiaccia V2/V3
        # Dropout 0.1 sulla scorciatoia res_ln_h1 per non monopolizzare lm_head
        self.res_h1_dropout = 0.0 if self.v1_only else float(res_h1_dropout)
        self.res_ln_h1 = nn.LayerNorm(d_model)
        self.res_drop_h1 = nn.Dropout(self.res_h1_dropout)
        if not self.v1_only:
            self.res_ln_h2 = nn.LayerNorm(d_model)
            self.res_ln_h3 = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_pezzi, bias=True)
        self.lm_head.weight = self.emb_v1.weight  # tied weight; bias separato

        self.register_buffer(
            "centroids",
            torch.from_numpy(self.mesh.face_centroids.astype("float32")),
        )
        self.register_buffer(
            "adjacency",
            torch.from_numpy(self.mesh.adjacency.astype("int64")),
        )

        # v6 dopo tutti i moduli: stessi valori iniziali di v5 per i parametri comuni a parità di seed
        self.hash_routing = bool(hash_routing)
        self.router_v6 = bool(router_v6) or self.hash_routing
        self.no_walk = bool(no_walk)
        self.sheet_mix = bool(sheet_mix)
        self.sheet_m = int(sheet_m)
        # v19: hop≥1 solo entry; False = storico v13–v18 (mean Δ su fogli cross-face)
        self.sheet_entry_only = bool(sheet_entry_only) and self.sheet_mix
        # v13 riusa il path sector_mean con buffer (F, 3m) al posto dello spicchio
        self.sector_mix = bool(sector_mix) or self.sheet_mix
        self.sector_frac = float(sector_frac)
        self.edge_neighbor_mix = bool(edge_neighbor_mix)
        self.edge_nb_tau = float(edge_nb_tau)
        # v11 sector-mix / v12 edge-nb / v13 sheet-mix implicano Spoke-to-Hub
        self.walk_to_center = (
            bool(walk_to_center) or self.sector_mix or self.edge_neighbor_mix
        )
        self.k_core = int(k_core)
        self.center_mix = float(center_mix)
        if self.sheet_mix and self.edge_neighbor_mix:
            raise ValueError("sheet_mix (v13) e edge_neighbor_mix (v12) sono mutuamente esclusivi")
        if self.sector_mix and self.edge_neighbor_mix:
            raise ValueError("sector_mix (v11) e edge_neighbor_mix (v12) sono mutuamente esclusivi")
        if self.sheet_mix and self.sheet_m < 1:
            raise ValueError("sheet_m deve essere >= 1")
        if bool(sheet_entry_only) and not self.sheet_mix:
            raise ValueError("sheet_entry_only richiede sheet_mix")
        if self.no_walk and self.walk_to_center:
            raise ValueError("no_walk e walk_to_center (Spoke-to-Hub) sono mutuamente esclusivi")
        if self.no_walk and (not self.router_v6 or int(self.walk.max_hops) != 1):
            raise ValueError("no_walk: servono router_v6 o hash_routing e WalkConfig(max_hops=1)")
        self.spoke_gates: Optional[nn.ModuleList] = None
        if self.walk_to_center:
            if not self.hash_routing:
                raise ValueError("walk_to_center/Spoke-to-Hub: serve hash_routing (ingresso Zipf)")
            hops = int(self.walk.max_hops)
            if hops < 2 or hops > 3:
                raise ValueError("walk_to_center/Spoke-to-Hub: H_max deve essere 2 o 3")
            if not (0.0 <= self.center_mix <= 1.0):
                raise ValueError("center_mix deve essere in [0, 1]")
            if self.sector_mix and not self.sheet_mix and not (0.0 < self.sector_frac <= 1.0):
                raise ValueError("sector_frac deve essere in (0, 1]")
            parent, is_hub, hub_ids = build_spoke_hub_parent(
                self.mesh.adjacency, self.mesh.face_centroids, k_core=self.k_core
            )
            self.register_buffer("parent_face", torch.from_numpy(parent.astype("int64")))
            self.register_buffer("is_hub", torch.from_numpy(is_hub.astype("bool")))
            self.register_buffer("hub_ids", torch.from_numpy(hub_ids.astype("int64")))
            if self.sheet_mix and self.sheet_entry_only:
                # v19: niente buffer fogli — hop≥1 tocca solo l'entry (vedi walk)
                self.sector_faces = None  # type: ignore[assignment]
                self.sector_k = 1
            elif self.sheet_mix:
                sheets = build_complementary_sheets(
                    self.mesh.face_centroids,
                    spiral_rank=getattr(self.mesh, "spiral_rank", None),
                    m_per_sheet=self.sheet_m,
                )
                self.register_buffer(
                    "sector_faces", torch.from_numpy(sheets.astype("int64"))
                )
                self.sector_k = int(sheets.shape[1])
            elif self.sector_mix:
                sectors = build_azimuth_sectors(
                    self.mesh.face_centroids, sector_frac=self.sector_frac
                )
                self.register_buffer(
                    "sector_faces", torch.from_numpy(sectors.astype("int64"))
                )
                self.sector_k = int(sectors.shape[1])
            else:
                self.sector_faces = None  # type: ignore[assignment]
                self.sector_k = 0
            if self.edge_neighbor_mix:
                w = build_edge_neighbor_weights(
                    self.mesh.adjacency, self.mesh.face_centroids, tau=self.edge_nb_tau
                )
                self.register_buffer("edge_nb_w", torch.from_numpy(w))
            else:
                self.edge_nb_w = None  # type: ignore[assignment]
            self.spoke_res_scale = 1.0 / math.sqrt(float(hops))
            # gate scalare hop≥1; bias=0 → σ≈0.5 a init (Argon)
            n_banks = 1 if self.v1_only else 3
            gates = []
            for _ in range(n_banks):
                lin = nn.Linear(self.d_model, 1)
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
                gates.append(lin)
            self.spoke_gates = nn.ModuleList(gates)
        else:
            self.parent_face = None  # type: ignore[assignment]
            self.is_hub = None  # type: ignore[assignment]
            self.hub_ids = None  # type: ignore[assignment]
            self.sector_faces = None  # type: ignore[assignment]
            self.sector_k = 0
            self.sector_mix = False
            self.sheet_mix = False
            self.sheet_entry_only = False
            self.edge_neighbor_mix = False
            self.edge_nb_w = None  # type: ignore[assignment]
            self.spoke_res_scale = 1.0
            self.center_mix = 0.0
        self.v6_logit_scale_max = V6_LOGIT_SCALE_MAX
        self.v6_bias_rate = V6_BIAS_RATE
        if self.router_v6:
            n_faces = int(self.mesh.num_faces)
            sizes = (self.vocab_pezzi, self.vocab_parole, self.vocab_unioni)
            for s, bank in enumerate(self._banks(), start=1):
                if self.hash_routing:
                    bank.register_buffer("hash_face", hash_faces(sizes[s - 1], n_faces, s))
                    delattr(self, f"to_sphere{s}")
                    setattr(self, f"to_sphere{s}", None)
                else:
                    bank.logit_scale = nn.Parameter(torch.tensor(V6_LOGIT_SCALE_INIT))
                    bank.register_buffer("face_bias", torch.zeros(n_faces))

        self.multi_slot_v1 = bool(multi_slot_v1)
        if zipf_hash_path is not None:
            if not self.hash_routing:
                raise ValueError("zipf_hash_path richiede hash_routing")
            self.load_zipf_hash(zipf_hash_path)
        elif self.multi_slot_v1:
            # Eval senza zipf path: buffer placeholder (V, K=3) riempito da state_dict
            v = int(self.bank1.hash_face.numel())
            self.bank1.register_buffer(
                "hash_slots",
                self.bank1.hash_face.unsqueeze(1).expand(v, 3).contiguous().clone(),
            )

        # piramidi e centri per ultimi: i parametri della base hanno gli stessi valori iniziali a parità di seed
        self.hex_pyramid = bool(hex_pyramid)
        self.shared_edges_verts = bool(shared_edges_verts)
        if self.hex_pyramid and self.shared_edges_verts:
            raise ValueError("hex_pyramid e shared_edges_verts (v12) sono mutuamente esclusivi")
        self.pyramid_center = bool(pyramid_center) or self.hex_pyramid or self.shared_edges_verts
        self.pyramids: Optional[nn.ModuleList] = None
        if self.pyramid_center:
            if not self.hash_routing:
                raise ValueError("pyramid_center/hex_pyramid: serve hash_routing (la piramide è quella della faccia d'ingresso)")
            self.pyramid_narrow = int(pyramid_narrow or self.d_model // 2)
            self.pyramid_hidden = int(pyramid_hidden or 2 * self.d_model)
            # generatore proprio e stato globale ripristinato: su CPU anche le maschere di dropout restano quelle della base
            gen = torch.Generator().manual_seed((torch.initial_seed() + PYRAMID_SEED_OFFSET) % 2**63)
            rng = torch.get_rng_state()
            if self.hex_pyramid:
                face_verts = torch.from_numpy(self.mesh.faces.astype("int64"))
                n_vert = int(self.mesh.num_vertices)
                self.pyramids = nn.ModuleList(
                    HexPyramidCenter(
                        self.d_model, n_vert, face_verts, self.pyramid_narrow, self.pyramid_hidden, gen
                    )
                    for _ in self._banks()
                )
            elif self.shared_edges_verts:
                face_edges_np, n_edges = build_face_edge_ids(self.mesh.faces)
                face_edges = torch.from_numpy(face_edges_np.astype("int64"))
                face_verts = torch.from_numpy(self.mesh.faces.astype("int64"))
                n_vert = int(self.mesh.num_vertices)
                self.pyramids = nn.ModuleList(
                    SharedEdgeVertexPyramid(
                        self.d_model,
                        int(self.mesh.num_faces),
                        int(n_edges),
                        n_vert,
                        face_edges,
                        face_verts,
                        self.pyramid_narrow,
                        self.pyramid_hidden,
                        gen,
                    )
                    for _ in self._banks()
                )
            else:
                self.pyramids = nn.ModuleList(
                    PyramidCenter(self.d_model, int(self.mesh.num_faces), self.pyramid_narrow, self.pyramid_hidden, gen)
                    for _ in self._banks()
                )
            torch.set_rng_state(rng)

    def _banks(self) -> List[PairedFaceBank]:
        return [self.bank1] if self.v1_only else [self.bank1, self.bank2, self.bank3]

    def load_zipf_hash(self, path: Union[str, Path]) -> None:
        """Copia ``hash_face_{1,2,3}`` (e opz. ``hash_slots_1`` multi-slot V1) da un ``.pt`` Zipf."""
        path = Path(path)
        try:
            blob = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            blob = torch.load(path, map_location="cpu")
        n_faces = int(self.mesh.num_faces)
        for s, bank in enumerate(self._banks(), start=1):
            key = f"hash_face_{s}"
            if key not in blob:
                raise KeyError(f"{path}: manca {key}")
            t = blob[key].long().view(-1)
            if t.numel() != bank.hash_face.numel():
                raise ValueError(
                    f"{path} {key}: len={t.numel()} ≠ bank.hash_face={bank.hash_face.numel()}"
                )
            if int(t.max()) >= n_faces or int(t.min()) < 0:
                raise ValueError(f"{path} {key}: facce fuori da [0, {n_faces})")
            bank.hash_face.copy_(t)
        # Multi-slot solo stadio 1 (pezzi): (V, K) con K copie della primaria se non top-N
        if "hash_slots_1" in blob:
            slots = blob["hash_slots_1"].long()
            if slots.ndim != 2 or slots.shape[0] != self.bank1.hash_face.numel():
                raise ValueError(
                    f"{path} hash_slots_1: shape={tuple(slots.shape)} "
                    f"atteso (V={self.bank1.hash_face.numel()}, K)"
                )
            if int(slots.max()) >= n_faces or int(slots.min()) < 0:
                raise ValueError(f"{path} hash_slots_1: facce fuori da [0, {n_faces})")
            self.bank1.register_buffer("hash_slots", slots.contiguous())
            self.multi_slot_v1 = True
        elif self.multi_slot_v1:
            raise ValueError(f"{path}: multi_slot_v1 richiesto ma manca hash_slots_1")

    def _active_mask(self, stage: int, ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Maschera token attivi per stadi 2/3 con ``mask_pad_stages`` (pad/unk → False)."""
        if not self.mask_pad_stages or stage < 1:
            return None
        if stage == 1:
            pad, unk = self.pad_id_v2, self.unk_id_v2
        elif stage == 2:
            pad, unk = self.pad_id_v3, self.unk_id_v3
        else:
            return None
        return (ids != pad) & (ids != unk)

    def _v6_entry(
        self,
        h: torch.Tensor,
        bank: PairedFaceBank,
        to_sphere: Optional[nn.Linear],
        entry_ids: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Faccia d'ingresso v6 (B,) e peso straight-through w (B, 1), None senza gradiente.

        w = p_f / sg(p_f) è scritto come exp(log p_f − sg(log p_f)): il bias può
        scegliere una faccia con p_f sotto il minimo di float32 e il rapporto
        diventerebbe 0/0. In training il carico d'ingresso per faccia si accumula
        sul device per ``update_face_bias``.
        """
        if self.hash_routing:
            if entry_ids is None:
                raise ValueError("hash_routing: servono gli id dei token per la faccia d'ingresso")
            # v9 multi-slot V1: in training campiona tra K facce; in eval usa lo slot 0 (primaria)
            slots = getattr(bank, "hash_slots", None)
            if slots is not None:
                row = slots[entry_ids]  # (..., K)
                if self.training:
                    k = int(row.shape[-1])
                    idx = torch.randint(0, k, entry_ids.shape, device=entry_ids.device)
                    face = row.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
                else:
                    face = row[..., 0]
                return face, None
            return bank.hash_face[entry_ids], None
        scores = F.normalize(to_sphere(h), dim=-1) @ self.centroids.T  # (B, F)
        face = (scores + bank.face_bias).argmax(dim=-1)
        if self.training:
            n_faces = scores.shape[1]
            load = (face.unsqueeze(1) == torch.arange(n_faces, device=face.device)).sum(0)
            prev = bank.__dict__.get("_v6_load")
            bank.__dict__["_v6_load"] = load if prev is None else prev + load
        if not torch.is_grad_enabled():
            return face, None
        scale = bank.logit_scale.clamp(max=self.v6_logit_scale_max).exp()
        lp = F.log_softmax(scores * scale, dim=-1).gather(1, face.unsqueeze(1))
        return face, torch.exp(lp - lp.detach())

    def _v6_route(
        self, face: torch.Tensor, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Porte come ``_route_step``, uscita dura a soglia; ritorna anche il peso del salto successivo.

        w = w_porta · p_stay / sg(p_stay) (B, 1), None senza gradiente: w_porta è il
        valore Gumbel-ST della porta scelta (solo in training), p_stay =
        1 − probs[:, 3] = exp(LSE(z[:3]) − LSE(z)). Sui token che proseguono il
        logit di uscita riceve d log p_stay / d z_uscita = −p_exit volte il
        gradiente del peso dell'esperto del salto successivo.
        Con ``no_walk`` escono tutti: porte e uscita non vengono valutate.
        Con ``walk_to_center`` (Spoke-to-Hub) il routing è in ``_walk_token_spoke_hub``:
        qui si usa solo ``parent_face`` (niente Gumbel).
        """
        if self.no_walk:
            return face, torch.ones_like(face, dtype=torch.bool), None
        if self.walk_to_center:
            nxt = self.parent_face[face.long()]
            exit_mask = nxt == face
            return nxt, exit_mask, None
        temp = max(self.walk.router_temperature, 1e-6)
        z = logits / temp
        probs = F.softmax(z, dim=-1)
        exit_mask = probs[:, 3] >= self.walk.exit_threshold
        neighbors = self.adjacency[face.long()]
        w_side: Optional[torch.Tensor] = None
        if self._hard_routing():
            side = probs[:, :3].argmax(dim=-1)
        else:
            side_oh = F.gumbel_softmax(logits[:, :3] / temp, hard=True, dim=-1)
            side = side_oh.argmax(dim=-1)
            w_side = side_oh.gather(1, side.unsqueeze(1))
        next_face = neighbors.gather(1, side.unsqueeze(1)).squeeze(1)
        if not torch.is_grad_enabled():
            return next_face, exit_mask, None
        log_stay = torch.logsumexp(z[:, :3], dim=-1, keepdim=True) - torch.logsumexp(z, dim=-1, keepdim=True)
        w = torch.exp(log_stay - log_stay.detach())
        return next_face, exit_mask, (w if w_side is None else w_side * w)

    def _walk_token_v6_loop(
        self,
        h: torch.Tensor,
        bank: PairedFaceBank,
        to_sphere: Optional[nn.Linear],
        entry_ids: Optional[torch.Tensor] = None,
        record_path: bool = False,
        active: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[List[int]]]]:
        """Cammino v6 col loop per faccia: riferimento di ``fast_walk_token_v6`` nei test."""
        b = h.shape[0]
        face, w = self._v6_entry(h, bank, to_sphere, entry_ids)
        if active is None:
            active = torch.ones(b, dtype=torch.bool, device=h.device)
        else:
            active = active.to(device=h.device, dtype=torch.bool)
        face0 = torch.where(active, face, torch.full_like(face, -1))
        paths = [[int(f)] for f in face0.tolist()] if record_path else None
        for _ in range(self.walk.max_hops):
            if not bool(active.any()):
                break
            logits = h.new_zeros(b, 4)
            h_new = h.clone()
            for f_id in face[active].unique().tolist():
                mask = active & (face == f_id)
                blk = bank.blocks[int(bank.face_to_slot[f_id].item())]
                x = h[mask]
                delta = blk.ff(blk.norm(x))
                out = x + delta if w is None else x + w[mask] * delta
                h_new[mask] = out
                logits[mask] = blk.router(out)
            h = h_new
            next_face, exit_mask, w = self._v6_route(face, logits)
            moved = active & ~exit_mask
            face = torch.where(moved, next_face, face)
            if paths is not None:
                for i in range(b):
                    if moved[i]:
                        paths[i].append(int(face[i].item()))
                    elif active[i] and exit_mask[i]:
                        paths[i].append(-1)
            active = active & ~exit_mask
        return h, paths

    @torch.no_grad()
    def update_face_bias(self) -> None:
        """Loss-Free Balancing: b_f += u · sign(carico medio − carico_f) col carico d'ingresso accumulato."""
        if not self.router_v6 or self.hash_routing:
            return
        for bank in self._banks():
            load = bank.__dict__.pop("_v6_load", None)
            if load is None:
                continue
            load = load.to(bank.face_bias.dtype)
            bank.face_bias.add_(torch.sign(load.mean() - load), alpha=self.v6_bias_rate)

    def _hard_routing(self) -> bool:
        """Hard (argmax) in eval o se WalkConfig.hard_routing; soft/ST in train."""
        return bool(self.walk.hard_routing) or (not self.training)

    def project_to_face(
        self, h: torch.Tensor, to_sphere: nn.Linear
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Proietta h sulla sfera → faccia.
        Train (soft): Gumbel-ST one-hot (B, F) così i grad arrivano a to_sphere.
        Eval/hard: solo face id (argmax), face_oh=None.
        """
        direction = F.normalize(to_sphere(h), dim=-1)
        scores = direction @ self.centroids.T  # (B, F)
        if self._hard_routing():
            return scores.argmax(dim=-1), None
        temp = max(self.walk.router_temperature, 1e-6)
        face_oh = F.gumbel_softmax(scores / temp, hard=True, dim=-1)  # ST soft→hard
        face = face_oh.argmax(dim=-1)  # solo indice forward; massa ST resta in face_oh
        return face, face_oh

    def _route_step(
        self, face: torch.Tensor, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Router ternario + exit.
        Train (soft): Gumbel-ST su 3 lati (niente argmax sulla one-hot) + mixture
        su adjacency → side_oh per propagare i grad al router.
        Eval/hard: argmax discreto, side_oh=None.
        """
        temp = max(self.walk.router_temperature, 1e-6)
        probs = F.softmax(logits / temp, dim=-1)
        exit_prob = probs[:, 3]
        neighbors = self.adjacency[face.long()]  # (B, 3)
        exit_mask = exit_prob >= self.walk.exit_threshold

        if self._hard_routing():
            side = probs[:, :3].argmax(dim=-1)
            next_face = neighbors.gather(1, side.unsqueeze(1)).squeeze(1)
            return next_face, exit_mask, None

        # Gumbel-ST: forward one-hot, backward softmax — non chiamare .argmax sulla ST
        side_oh = F.gumbel_softmax(logits[:, :3] / temp, hard=True, dim=-1)  # (B, 3)
        side = side_oh.argmax(dim=-1)
        next_face = neighbors.gather(1, side.unsqueeze(1)).squeeze(1)
        return next_face, exit_mask, side_oh

    def _scatter_side_oh(
        self, face: torch.Tensor, side_oh: torch.Tensor
    ) -> torch.Tensor:
        """Soft/ST mixture su adjacency: massa dei 3 lati → one-hot sparsa su F facce."""
        b = face.shape[0]
        n_faces = int(self.centroids.shape[0])
        neighbors = self.adjacency[face.long()]  # (B, 3)
        face_oh = side_oh.new_zeros(b, n_faces)
        face_oh.scatter_add_(1, neighbors, side_oh)
        return face_oh

    def _spoke_stage_index(self, bank: PairedFaceBank) -> int:
        for i, b in enumerate(self._banks()):
            if b is bank:
                return i
        raise KeyError("bank non appartiene al modello")

    def _apply_sector_mean(
        self,
        h: torch.Tensor,
        bank: PairedFaceBank,
        entry: torch.Tensor,
        active: torch.Tensor,
        g: Optional[torch.Tensor],
        st: Optional[dict] = None,
    ) -> torch.Tensor:
        """v11: ``x ← x + g · mean_{f∈settore(entry)}(Δ_f)`` (hop≥1).

        Settore fissato sulla faccia d'ingresso Zipf. ``g=None`` → peso 1.0.
        """
        assert self.sector_faces is not None
        k = int(self.sector_k)
        if k <= 0:
            return h
        use_fast = st is not None
        delta_sum: Optional[torch.Tensor] = None
        for slot in range(k):
            face_k = self.sector_faces[entry.long(), slot]
            if use_fast:
                assert st is not None
                out, _lg, counts = grouped_block_forward(
                    st, h, face_k, active, gate=None
                )
                if st["visited"] is not None:
                    st["visited"].add(counts)  # type: ignore[union-attr]
                # out = x+Δ sulle attive; sulle inattive out è spazzatura → maschera dopo
                d = out - h
            else:
                d = h.new_zeros(h.shape)
                for f_id in face_k[active].unique().tolist():
                    mask = active & (face_k == f_id)
                    blk = bank.blocks[int(bank.face_to_slot[f_id].item())]
                    x = h[mask]
                    d[mask] = blk.ff(blk.norm(x))
            delta_sum = d if delta_sum is None else delta_sum + d
        assert delta_sum is not None
        mean_d = delta_sum / float(k)
        if g is None:
            upd = h + mean_d
        else:
            upd = h + g * mean_d
        return torch.where(active.unsqueeze(-1), upd, h)

    def _apply_edge_neighbor_mix(
        self,
        h: torch.Tensor,
        bank: PairedFaceBank,
        entry: torch.Tensor,
        active: torch.Tensor,
        g: Optional[torch.Tensor],
        st: Optional[dict] = None,
    ) -> torch.Tensor:
        """v12: ``x ← x + g · Σ_k w_{e,k} Δ_{nb_k}`` sui 3 lati di ``entry`` (pesi verso polo)."""
        assert self.edge_nb_w is not None
        e = entry.long()
        nbs = self.adjacency[e]  # (B, 3)
        w = self.edge_nb_w[e]  # (B, 3)
        use_fast = st is not None
        weighted: Optional[torch.Tensor] = None
        for slot in range(3):
            face_k = nbs[:, slot]
            wk = w[:, slot].unsqueeze(-1)
            if use_fast:
                assert st is not None
                out, _lg, counts = grouped_block_forward(
                    st, h, face_k, active, gate=None
                )
                if st["visited"] is not None:
                    st["visited"].add(counts)  # type: ignore[union-attr]
                d = out - h
            else:
                d = h.new_zeros(h.shape)
                for f_id in face_k[active].unique().tolist():
                    mask = active & (face_k == f_id)
                    blk = bank.blocks[int(bank.face_to_slot[f_id].item())]
                    x = h[mask]
                    d[mask] = blk.ff(blk.norm(x))
            term = wk * d
            weighted = term if weighted is None else weighted + term
        assert weighted is not None
        if g is None:
            upd = h + weighted
        else:
            upd = h + g * weighted
        return torch.where(active.unsqueeze(-1), upd, h)

    def _walk_token_spoke_hub(
        self,
        h: torch.Tensor,
        bank: PairedFaceBank,
        to_sphere: Optional[nn.Linear],
        entry_ids: Optional[torch.Tensor] = None,
        record_path: bool = False,
        active: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[List[int]]]]:
        """v10 Spoke-to-Hub / v11 sector-mix / v12 edge-nb / v13–v19 sheet: parent→hub; no Gumbel.

        hop0: solo faccia entry, residuo pieno.
        hop≥1: v10 = blocco ``face`` corrente con ``g=(c+(1-c)σ)/√H``;
        v11 (``sector_mix``) = media Δ sul settore azimutale dell'ingresso;
        v12 (``edge_neighbor_mix``) = mix pesato sui 3 vicini di lato (anche al polo);
        v13–v18 (``sheet_mix``, default) = media Δ sui fogli complementari dell'ingresso;
        v19 (``sheet_entry_only``) = di nuovo solo blocco entry (niente cross-face).
        pad/unk (``active=False``): 0 hop. Train==eval.
        """
        assert self.spoke_gates is not None and self.parent_face is not None
        b = h.shape[0]
        face, _ = self._v6_entry(h, bank, to_sphere, entry_ids)
        if active is None:
            active = torch.ones(b, dtype=torch.bool, device=h.device)
        else:
            active = active.to(device=h.device, dtype=torch.bool)
        entry = face.clone()
        face0 = torch.where(active, face, torch.full_like(face, -1))
        max_hops = int(self.walk.max_hops)
        if b == 0 or max_hops <= 0:
            return h, (_build_paths(face0, []) if record_path else None)

        gate_lin = self.spoke_gates[self._spoke_stage_index(bank)]
        scale = float(self.spoke_res_scale)
        codes: List[torch.Tensor] = []
        use_fast = bool(self.fast_walk)
        st = stack_bank(bank) if use_fast else None
        # storico sheet/sector: media su sector_faces; v19 entry-only salta questo path
        use_sector = (
            bool(self.sector_mix)
            and self.sector_faces is not None
            and not bool(self.sheet_entry_only)
        )
        use_sheet_entry = bool(self.sheet_mix) and bool(self.sheet_entry_only)
        use_edge_nb = bool(self.edge_neighbor_mix) and self.edge_nb_w is not None

        c_mix = float(self.center_mix)
        for hop in range(max_hops):
            if not bool(active.any()):
                break
            # hop0: residuo pieno (v8.1); hop≥1: g=(c+(1-c)σ)/√H  (liscio, no clamp)
            if hop == 0:
                g: Optional[torch.Tensor] = None
            else:
                g = (c_mix + (1.0 - c_mix) * torch.sigmoid(gate_lin(h))) * scale
            if use_edge_nb and hop >= 1:
                h = self._apply_edge_neighbor_mix(h, bank, entry, active, g, st=st)
            elif use_sector and hop >= 1:
                h = self._apply_sector_mean(h, bank, entry, active, g, st=st)
            elif use_sheet_entry and hop >= 1:
                # v19: solo esperto della faccia d'ingresso (fogli non toccano altre facce)
                apply_face = entry
                if use_fast:
                    assert st is not None
                    out, _lg, counts = grouped_block_forward(
                        st, h, apply_face, active, gate=g
                    )
                    if st["visited"] is not None:
                        st["visited"].add(counts)  # type: ignore[union-attr]
                    h = torch.where(active.unsqueeze(-1), out, h)
                else:
                    h_new = h.clone()
                    for f_id in apply_face[active].unique().tolist():
                        mask = active & (apply_face == f_id)
                        blk = bank.blocks[int(bank.face_to_slot[f_id].item())]
                        x = h[mask]
                        delta = blk.ff(blk.norm(x))
                        if g is None:
                            h_new[mask] = x + delta
                        else:
                            h_new[mask] = x + g[mask] * delta
                    h = h_new
            elif use_fast:
                assert st is not None
                out, _lg, counts = grouped_block_forward(
                    st, h, face, active, gate=g
                )
                if st["visited"] is not None:
                    st["visited"].add(counts)  # type: ignore[union-attr]
                h = torch.where(active.unsqueeze(-1), out, h)
            else:
                h_new = h.clone()
                for f_id in face[active].unique().tolist():
                    mask = active & (face == f_id)
                    blk = bank.blocks[int(bank.face_to_slot[f_id].item())]
                    x = h[mask]
                    delta = blk.ff(blk.norm(x))
                    if g is None:
                        h_new[mask] = x + delta
                    else:
                        h_new[mask] = x + g[mask] * delta
                h = h_new
            # v12: nessuno stop anticipato su hub — tutti fanno hop1 sui 3 lati
            if use_edge_nb:
                stop = torch.full(
                    (b,), hop == max_hops - 1, device=h.device, dtype=torch.bool
                )
                moved = torch.zeros_like(active)
                # face resta sull'entry; hop1 usa adjacency[entry]
            else:
                nxt = self.parent_face[face.long()]
                at_hub = nxt == face
                stop = at_hub | (hop == max_hops - 1)
                moved = active & ~stop
                face = torch.where(moved, nxt, face)
            just_exit = active & stop
            active = active & ~stop
            if record_path:
                codes.append(
                    torch.where(
                        moved,
                        face,
                        torch.where(just_exit, torch.full_like(face, -1), torch.full_like(face, -2)),
                    )
                )

        return h, (_build_paths(face0, codes) if record_path else None)

    def walk_token(
        self,
        h: torch.Tensor,
        bank: PairedFaceBank,
        to_sphere: Optional[nn.Linear],
        record_path: bool = False,
        entry_ids: Optional[torch.Tensor] = None,
        active: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[List[int]]]]:
        if self.walk_to_center:
            return self._walk_token_spoke_hub(
                h, bank, to_sphere, entry_ids=entry_ids, record_path=record_path, active=active
            )
        if self.router_v6:
            if self.fast_walk:
                return fast_walk_token_v6(
                    self, h, bank, to_sphere, entry_ids=entry_ids, record_path=record_path, active=active
                )
            return self._walk_token_v6_loop(
                h, bank, to_sphere, entry_ids=entry_ids, record_path=record_path, active=active
            )
        if self.fast_walk:
            return fast_walk_token(self, h, bank, to_sphere, record_path=record_path)
        b = h.shape[0]
        face, face_oh = self.project_to_face(h, to_sphere)
        paths: Optional[List[List[int]]] = (
            [[int(face[i].item())] for i in range(b)] if record_path else None
        )
        active = torch.ones(b, dtype=torch.bool, device=h.device)

        for _ in range(self.walk.max_hops):
            if not active.any():
                break
            logits = h.new_zeros(b, 4)
            h_new = h.clone()
            for f_id in face[active].unique().tolist():
                mask = active & (face == f_id)
                if not mask.any():
                    continue
                out, lg = bank.forward_face(int(f_id), h[mask])
                if face_oh is not None:
                    # ST: w≈1 in forward; backward → face_oh → to_sphere / router
                    w = face_oh[mask, int(f_id)].unsqueeze(-1)
                    h_new[mask] = out * w
                else:
                    h_new[mask] = out
                logits[mask] = lg
            h = h_new
            next_face, exit_mask, side_oh = self._route_step(face, logits)
            moved = active & ~exit_mask
            if side_oh is not None:
                # sum(ST one-hot)=1 in forward: ancora i grad al router anche su exit/ultimo hop
                scale = side_oh.sum(dim=-1, keepdim=True)
                h = torch.where(active.unsqueeze(-1), h * scale, h)
                routed_oh = self._scatter_side_oh(face, side_oh)
                if face_oh is None:
                    face_oh = routed_oh
                else:
                    face_oh = torch.where(moved.unsqueeze(-1), routed_oh, face_oh)
            face = torch.where(moved, next_face, face)
            just_exit = active & exit_mask
            active = active & ~exit_mask
            if record_path and paths is not None:
                for i in range(b):
                    if moved[i]:
                        paths[i].append(int(face[i].item()))
                    elif just_exit[i]:
                        paths[i].append(-1)
        return h, paths

    def _descend(
        self,
        stage: int,
        h: torch.Tensor,
        ids: torch.Tensor,
        active_bt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Piramide della faccia d'ingresso e centro dello stadio (da 0) su h (B, T, d); identità senza piramidi."""
        if self.pyramids is None:
            return h
        bank = self._banks()[stage]
        face = bank.hash_face[ids]
        # Allinea multi-slot V1 alla stessa regola di ``_v6_entry`` (slot 0 in eval; sample in train)
        slots = getattr(bank, "hash_slots", None)
        if slots is not None:
            row = slots[ids]
            if self.training:
                k = int(row.shape[-1])
                idx = torch.randint(0, k, ids.shape, device=ids.device)
                face = row.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            else:
                face = row[..., 0]
        out = self.pyramids[stage](h, face)
        if active_bt is None:
            return out
        return torch.where(active_bt.unsqueeze(-1), out, h)

    def _walk_seq(
        self,
        h: torch.Tensor,
        bank: PairedFaceBank,
        to_sphere: Optional[nn.Linear],
        record_path: bool,
        entry_ids: Optional[torch.Tensor] = None,
        active_bt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[List[List[int]]]]]:
        b, t, d = h.shape
        kw: dict = {}
        if entry_ids is not None:
            kw["entry_ids"] = entry_ids.reshape(b * t)
        if active_bt is not None:
            kw["active"] = active_bt.reshape(b * t)
        flat, paths_flat = self.walk_token(
            h.reshape(b * t, d), bank, to_sphere, record_path=record_path, **kw
        )
        h_out = flat.view(b, t, d)
        paths = None
        if record_path and paths_flat is not None:
            paths = [
                [paths_flat[bi * t + ti] for ti in range(t)] for bi in range(b)
            ]
        return h_out, paths

    def forward(
        self,
        pezzi_ids: torch.Tensor,
        parole_ids: Optional[torch.Tensor] = None,
        unioni_ids: Optional[torch.Tensor] = None,
        record_path: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[List[List[int]]]]]:
        """
        pezzi/parole/unioni: (B, T) allineati sulla timeline pezzi.
        Se self.v1_only: richiede solo pezzi_ids (parole_ids/unioni_ids ignorati/None).
        returns: logits (B, T, V_pezzi), paths opzionale dell'ultimo mesh walk.
        """
        b, t = pezzi_ids.shape
        if t > self.max_seq_len:
            raise ValueError(f"seq_len={t} > max_seq_len={self.max_seq_len}")

        pos = torch.arange(t, device=pezzi_ids.device).unsqueeze(0)
        h = self.emb_v1(pezzi_ids) * self.emb_scale + self.pos_emb(pos)

        # Dei CausalAttnBlock: 1 su V1 (prima di bank1), 1 dopo
        for block in self.attn_blocks[:-1]:
            h = block(h)

        hashed = self.hash_routing
        if self.v1_only:
            h1, paths = self._walk_seq(
                h, self.bank1, self.to_sphere1, record_path=record_path,
                entry_ids=pezzi_ids if hashed else None,
            )
            h1 = self._descend(0, h1, pezzi_ids)
            h = self.res_ln_h1(h1)
            if len(self.attn_blocks) > 0:
                h = self.attn_blocks[-1](h)
            h = self.final_norm(h)
            logits = self.lm_head(h)
            logits = 30.0 * torch.tanh(logits / 30.0)
            return logits, paths

        if parole_ids is None or unioni_ids is None:
            raise ValueError("Cascade 3-stadi richiede parole_ids e unioni_ids")

        h1, _ = self._walk_seq(
            h, self.bank1, self.to_sphere1, record_path=False, entry_ids=pezzi_ids if hashed else None
        )
        h1 = self._descend(0, h1, pezzi_ids)

        # Stage 2: fuse bilanciato (LN per-addendo) + emb parole (+ attn inter-stadio opzionale)
        # v8.1 fuse_mask_pad: su pad/unk azzera l'addendo emb (mask-pad già saltava solo walk/piramide)
        active2 = self._active_mask(1, parole_ids)
        emb2 = self.fuse_ln_e2(self.emb_v2(parole_ids))
        if self.fuse_mask_pad and active2 is not None:
            emb2 = emb2 * active2.unsqueeze(-1).to(dtype=emb2.dtype)
        h = self.fuse_norm2(self.fuse_ln_h2(self.proj_12(h1)) + emb2)
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

        # Stage 3: fuse bilanciato + emb unioni
        active3 = self._active_mask(2, unioni_ids)
        emb3 = self.fuse_ln_e3(self.emb_v3(unioni_ids))
        if self.fuse_mask_pad and active3 is not None:
            emb3 = emb3 * active3.unsqueeze(-1).to(dtype=emb3.dtype)
        h = self.fuse_norm3(self.fuse_ln_h3(self.proj_23(h2)) + emb3)
        h3, paths = self._walk_seq(
            h,
            self.bank3,
            self.to_sphere3,
            record_path=record_path,
            entry_ids=unioni_ids if hashed else None,
            active_bt=active3,
        )
        h3 = self._descend(2, h3, unioni_ids, active_bt=active3)

        # Residual bilanciato: norm per-stadio (con dropout 0.1 su res_ln_h1) + attn post-fuse
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


class V1OnlyMeshModel(CascadeMeshModel):
    """Solo V1 (pezzi): emb_v1 (+pos) → CausalAttn_0 → bank1 (80 facce) → LN → CausalAttn_1 → final_norm → lm_head."""

    def __init__(
        self,
        vocab_pezzi: int,
        d_model: int = 256,
        frequency: int = 2,
        walk: Optional[WalkConfig] = None,
        d_hidden: Optional[int] = None,
        n_attn_layers: int = 2,
        n_heads: Optional[int] = None,
        max_seq_len: int = 512,
        attn_dropout: float = 0.0,
        compass_spiral: bool = False,
        fast_walk: bool = False,
    ):
        super().__init__(
            vocab_pezzi=vocab_pezzi,
            vocab_parole=0,
            vocab_unioni=0,
            d_model=d_model,
            frequency=frequency,
            walk=walk,
            d_hidden=d_hidden,
            n_attn_layers=n_attn_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            res_h1_dropout=0.0,
            v1_only=True,
            compass_spiral=compass_spiral,
            fast_walk=fast_walk,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
