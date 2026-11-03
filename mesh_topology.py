"""
Mesh geodetica sferica a tasselli triangolari.

Parte da un icosaedro regolare e lo suddivide con frequenza `frequency`
(ogni faccia → frequency² triangoli). Espone:
  - indici faccia → 3 vertici
  - tabella di adiacenza: 3 vicini per faccia (ordine lati 0,1,2)
  - packing a coppie di facce adiacenti per GEMM rettangolari
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

_PHI = (1.0 + np.sqrt(5.0)) / 2.0


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def icosahedron_vertices() -> np.ndarray:
    """12 vertici icosaedro (ordine classico three.js / Wikipedia)."""
    verts = np.asarray(
        [
            [-1, _PHI, 0],
            [1, _PHI, 0],
            [-1, -_PHI, 0],
            [1, -_PHI, 0],
            [0, -1, _PHI],
            [0, 1, _PHI],
            [0, -1, -_PHI],
            [0, 1, -_PHI],
            [_PHI, 0, -1],
            [_PHI, 0, 1],
            [-_PHI, 0, -1],
            [-_PHI, 0, 1],
        ],
        dtype=np.float64,
    )
    return _normalize(verts)


def icosahedron_faces() -> np.ndarray:
    """20 facce triangolari."""
    return np.asarray(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int64,
    )


def _edge_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True)
class GeodesicMesh:
    frequency: int
    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3)
    adjacency: np.ndarray  # (F, 3) vicino oltre lato i
    face_centroids: np.ndarray  # (F, 3)
    pairs: np.ndarray  # (P, 2) packing GEMM
    unpaired: np.ndarray  # (U,)
    compass_spiral: bool = False
    spiral_order: Optional[np.ndarray] = None  # (F,) face_id ordinati lungo la spirale polo→polo
    spiral_rank: Optional[np.ndarray] = None  # (F,) rango 0..F-1 lungo la spirale per ogni face_id
    spiral_band: Optional[np.ndarray] = None  # (F,) fascia latitudinale 0..B-1 per ogni face_id

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])


def build_geodesic_mesh(
    frequency: int = 2,
    *,
    compass_spiral: bool = False,
) -> GeodesicMesh:
    """
    Mesh geodetica a frequenza `frequency` (>= 1).
    Faces = 20 * frequency².
    Se `compass_spiral=True`, riordina i 3 vicini esistenti di ogni faccia
    lungo la spirale da polo a polo:
      porta 0 = avanti lungo la spirale,
      porta 1 = indietro lungo la spirale,
      porta 2 = cambio di fascia (nord/sud).
    """
    if frequency < 1:
        raise ValueError("frequency deve essere >= 1")

    base_v = icosahedron_vertices()
    base_f = icosahedron_faces()

    vert_list: List[np.ndarray] = [base_v[i].copy() for i in range(len(base_v))]
    # chiave: (lo, hi, steps_from_lo) → index, steps in 1..freq-1
    edge_cache: Dict[Tuple[int, int, int], int] = {}

    def point_on_edge(i: int, j: int, steps_from_i: int) -> int:
        """steps_from_i=0 → i, =frequency → j."""
        if steps_from_i == 0:
            return i
        if steps_from_i == frequency:
            return j
        lo, hi = _edge_key(i, j)
        steps_from_lo = steps_from_i if i == lo else frequency - steps_from_i
        key = (lo, hi, steps_from_lo)
        if key in edge_cache:
            return edge_cache[key]
        t = steps_from_lo / float(frequency)
        p = _normalize((1.0 - t) * base_v[lo] + t * base_v[hi])
        idx = len(vert_list)
        vert_list.append(p)
        edge_cache[key] = idx
        return idx

    faces: List[Tuple[int, int, int]] = []

    for v0, v1, v2 in base_f:
        v0, v1, v2 = int(v0), int(v1), int(v2)
        grid: Dict[Tuple[int, int], int] = {}

        for a in range(frequency + 1):
            for b in range(frequency + 1 - a):
                c = frequency - a - b  # peso su v0
                # a verso v1, b verso v2
                if a == 0 and b == 0:
                    grid[(a, b)] = v0
                elif b == 0:
                    grid[(a, b)] = point_on_edge(v0, v1, a)
                elif a == 0:
                    grid[(a, b)] = point_on_edge(v0, v2, b)
                elif c == 0:
                    grid[(a, b)] = point_on_edge(v1, v2, b)
                else:
                    p = _normalize(
                        (c * base_v[v0] + a * base_v[v1] + b * base_v[v2]) / float(frequency)
                    )
                    idx = len(vert_list)
                    vert_list.append(p)
                    grid[(a, b)] = idx

        for a in range(frequency):
            for b in range(frequency - a):
                i00 = grid[(a, b)]
                i10 = grid[(a + 1, b)]
                i01 = grid[(a, b + 1)]
                faces.append((i00, i10, i01))
                if a + b + 1 < frequency:
                    i11 = grid[(a + 1, b + 1)]
                    faces.append((i10, i11, i01))

    vertices = _normalize(np.stack(vert_list, axis=0))
    faces_arr = np.asarray(faces, dtype=np.int64)
    expected = 20 * frequency * frequency
    if faces_arr.shape[0] != expected:
        raise RuntimeError(f"Attese {expected} facce, ottenute {faces_arr.shape[0]}")

    adjacency = build_adjacency(faces_arr)
    centroids = _normalize(vertices[faces_arr].mean(axis=1))
    spiral_adj, spiral_order, spiral_rank, spiral_band = reorder_adjacency_spiral_compass(
        adjacency, centroids
    )
    if compass_spiral:
        adjacency = spiral_adj
    pairs, unpaired = pack_face_pairs(adjacency)

    return GeodesicMesh(
        frequency=frequency,
        vertices=vertices,
        faces=faces_arr,
        adjacency=adjacency,
        face_centroids=centroids,
        pairs=pairs,
        unpaired=unpaired,
        compass_spiral=bool(compass_spiral),
        spiral_order=spiral_order,
        spiral_rank=spiral_rank,
        spiral_band=spiral_band,
    )


def reorder_adjacency_spiral_compass(
    adjacency: np.ndarray,
    face_centroids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ordina i centroidi lungo una spirale da polo a polo attorno all'asse polare
    icosaedrico (vertice 0 → vertice 3, simmetria 5-fold) e permuta i 3 vicini
    già presenti in `adjacency` per ogni faccia `f`:
      - porta 0: vicino «avanti» lungo la spirale (+Δlon sulla fascia),
      - porta 1: vicino «indietro» lungo la spirale (-Δlon sulla fascia),
      - porta 2: cambio di fascia (il terzo vicino, attraversamento nord/sud).

    Non aggiunge né rimuove archi o facce: per ogni f, set(new_adj[f]) == set(adjacency[f]).
    Restituisce (new_adj, spiral_order, spiral_rank, spiral_band).
    """
    n_faces = int(face_centroids.shape[0])
    if adjacency.shape != (n_faces, 3):
        raise ValueError(f"adjacency attesa ({n_faces}, 3), ricevuta {adjacency.shape}")

    # Asse polare 5-fold dell'icosaedro: polo nord = vertice 0, polo sud = vertice 3
    pole = icosahedron_vertices()[0]
    u3 = pole / np.linalg.norm(pole)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u1 = ref - np.dot(ref, u3) * u3
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(u3, u1)

    z = face_centroids @ u3  # latitudine coseno in [-1, +1] (polo nord = +1, polo sud = -1)
    lon = np.mod(np.arctan2(face_centroids @ u2, face_centroids @ u1), 2.0 * np.pi)

    new_adj = np.empty_like(adjacency)
    for f in range(n_faces):
        nbs = adjacency[f]
        # Differenza angolare ciclica in (-pi, +pi] rispetto alla faccia f
        dlons = [
            float(((lon[int(nb)] - lon[f] + np.pi) % (2.0 * np.pi)) - np.pi)
            for nb in nbs
        ]
        # Il vicino di cambio fascia (nord/sud) attraversa il lato orizzontale (|Δlon| minimo, |Δz| massimo);
        # gli altri due muovono lungo la fascia in senso +Δlon (avanti, porta 0) e -Δlon (indietro, porta 1).
        idx_band = int(np.argmin(np.abs(dlons)))
        rem = [i for i in range(3) if i != idx_band]
        if dlons[rem[0]] >= dlons[rem[1]]:
            idx_fwd, idx_bwd = rem[0], rem[1]
        else:
            idx_fwd, idx_bwd = rem[1], rem[0]
        new_adj[f, 0] = nbs[idx_fwd]
        new_adj[f, 1] = nbs[idx_bwd]
        new_adj[f, 2] = nbs[idx_band]

    # Costruisce l'ordine spirale da polo nord (+z) a polo sud (-z) seguendo i cicli di porta 0
    visited = np.zeros(n_faces, dtype=bool)
    bands: List[List[int]] = []
    for f in range(n_faces):
        if visited[f]:
            continue
        cyc: List[int] = []
        cur = f
        while not visited[cur]:
            visited[cur] = True
            cyc.append(cur)
            cur = int(new_adj[cur, 0])
        # Ruota il ciclo in modo che inizi dalla faccia con longitudine minima in [0, 2pi)
        start_idx = int(np.argmin([lon[idx] for idx in cyc]))
        cyc = cyc[start_idx:] + cyc[:start_idx]
        bands.append(cyc)

    # Ordina le fasce da polo nord (z massima) a polo sud (z minima)
    bands.sort(key=lambda cyc: -float(np.mean(z[cyc])))

    spiral_order_list: List[int] = []
    spiral_band = np.empty(n_faces, dtype=np.int64)
    for b_idx, cyc in enumerate(bands):
        for f_id in cyc:
            spiral_band[f_id] = b_idx
            spiral_order_list.append(f_id)

    spiral_order = np.asarray(spiral_order_list, dtype=np.int64)
    spiral_rank = np.empty(n_faces, dtype=np.int64)
    spiral_rank[spiral_order] = np.arange(n_faces, dtype=np.int64)

    return new_adj, spiral_order, spiral_rank, spiral_band


def build_adjacency(faces: np.ndarray) -> np.ndarray:
    """adjacency[f, i] = faccia oltre lato i; lato0=(v0,v1), lato1=(v1,v2), lato2=(v2,v0)."""
    fcount = faces.shape[0]
    edge_to_face: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for fi, (a, b, c) in enumerate(faces):
        for li, (u, v) in enumerate(((int(a), int(b)), (int(b), int(c)), (int(c), int(a)))):
            edge_to_face.setdefault(_edge_key(u, v), []).append((fi, li))

    adj = np.full((fcount, 3), -1, dtype=np.int64)
    for occ in edge_to_face.values():
        if len(occ) != 2:
            continue
        (f0, l0), (f1, l1) = occ
        adj[f0, l0] = f1
        adj[f1, l1] = f0

    if np.any(adj < 0):
        bad = int(np.sum(adj < 0))
        raise RuntimeError(f"Adiacenza incompleta ({bad} lati aperti)")
    return adj


def pack_face_pairs(adjacency: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Matching greedy di facce adiacenti → matrici 2×D contigue."""
    fcount = adjacency.shape[0]
    used = np.zeros(fcount, dtype=bool)
    pairs: List[Tuple[int, int]] = []
    for f in range(fcount):
        if used[f]:
            continue
        partner = -1
        for n in adjacency[f]:
            ni = int(n)
            if not used[ni]:
                partner = ni
                break
        if partner < 0:
            continue
        used[f] = True
        used[partner] = True
        pairs.append((f, partner) if f < partner else (partner, f))
    unpaired = np.asarray([i for i in range(fcount) if not used[i]], dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64), unpaired


def nearest_face(mesh: GeodesicMesh, directions: np.ndarray) -> np.ndarray:
    """Argmax allineamento con centroidi faccia (prodotto scalare su sfera)."""
    d = _normalize(np.asarray(directions, dtype=np.float64))
    scores = d @ mesh.face_centroids.T
    return np.argmax(scores, axis=-1).astype(np.int64)


def describe_mesh(mesh: GeodesicMesh) -> str:
    return (
        f"GeodesicMesh(freq={mesh.frequency}, V={mesh.num_vertices}, "
        f"F={mesh.num_faces}, pairs={len(mesh.pairs)}, unpaired={len(mesh.unpaired)})"
    )
