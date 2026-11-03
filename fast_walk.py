"""
Cammino vettorizzato sulla mesh per CascadeMeshModel / V1OnlyMeshModel.

Sostituisce il loop Python per faccia di ``CascadeMeshModel.walk_token``
(``for f_id in face[active].unique().tolist()`` + ``forward_face`` con
``.item()`` e maschere booleane) con un dispatch raggruppato alla MegaBlocks:

  - i parametri dei TriangleBlock sono impilati per face id (stack
    differenziabile dei parametri esistenti: state_dict e checkpoint invariati);
  - i token attivi sono disposti per faccia (posizioni da un cumsum, niente
    sort): le facce con pochi token in un elemento di batch per faccia, quelle
    affollate in blocchi da ``block_rows`` righe con i pesi raccolti per blocco
    (vedi ``grouped_block_forward`` per il perché dei percorsi);
  - ogni Linear del blocco diventa un ``baddbmm`` per percorso;
  - le uscite tornano ai token con ``index_select`` e ``torch.where``.

Le forme dipendono solo dal numero di token B, mai dai dati: nel salto non ci
sono ``unique``, ``tolist``, ``nonzero`` né ``.item()``.

Puro refactoring: proiezione sulla faccia, router, Gumbel-ST, soglia di uscita
e peso straight-through sull'uscita intera del blocco sono quelli del modello
(si chiamano gli stessi metodi ``project_to_face``, ``_route_step``,
``_scatter_side_oh`` nello stesso ordine), quindi anche il consumo del
generatore casuale è lo stesso.  Restano due sincronizzazioni, entrambe
necessarie per l'equivalenza:
  - ``active.any()`` a inizio salto con routing soft: il codice originale esce
    dal ciclo quando tutti i token sono usciti e da lì non estrae più rumore
    Gumbel (in routing hard i salti in più sono no-op e il controllo è saltato);
  - nel backward, una lettura per banca della maschera delle facce visitate:
    le facce mai visitate devono restare con grad None come nel loop.

``fast_walk_token_v6`` è il cammino del router v6 (vedi geodesic_stages_model):
stesso dispatch raggruppato, peso straight-through passato come ``gate`` sul
solo ramo dell'esperto.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_BLOCK_ROWS = 16  # >= 16: regime "M grande" del Linear su MPS
DEFAULT_DENSE_ROWS = 15  # <= 15: regime "M piccolo" del Linear su MPS

_NAMES = ("ln_w", "ln_b", "w1", "b1", "w2", "b2", "wr", "br")


def _check_block(blk: nn.Module) -> None:
    norm, ff, router = blk.norm, blk.ff, blk.router
    ok = (
        isinstance(norm, nn.LayerNorm)
        and norm.weight is not None
        and norm.bias is not None
        and len(norm.normalized_shape) == 1
        and isinstance(ff, nn.Sequential)
        and len(ff) == 3
        and isinstance(ff[0], nn.Linear)
        and isinstance(ff[1], nn.GELU)
        and ff[1].approximate == "none"
        and isinstance(ff[2], nn.Linear)
        and ff[0].bias is not None
        and ff[2].bias is not None
        and isinstance(router, nn.Linear)
        and router.bias is not None
    )
    if not ok:
        raise TypeError("fast_walk: TriangleBlock con struttura inattesa (LN → Linear → GELU → Linear, router)")


def _face_blocks(bank: nn.Module) -> List[nn.Module]:
    """TriangleBlock in ordine di face id (il ModuleList è in ordine di packing a coppie)."""
    slots = bank.__dict__.get("_fast_walk_slots")
    if slots is None:
        slots = [int(s) for s in bank.face_to_slot.detach().cpu().tolist()]
        if sorted(slots) != list(range(len(bank.blocks))) or len(slots) != int(bank.num_faces):
            raise ValueError("fast_walk: face_to_slot non è una permutazione dei blocchi")
        for s in slots:
            _check_block(bank.blocks[s])
        bank.__dict__["_fast_walk_slots"] = slots
    return [bank.blocks[s] for s in slots]


def _block_params(blk: nn.Module) -> Tuple[torch.Tensor, ...]:
    return (
        blk.norm.weight,
        blk.norm.bias,
        blk.ff[0].weight,
        blk.ff[0].bias,
        blk.ff[2].weight,
        blk.ff[2].bias,
        blk.router.weight,
        blk.router.bias,
    )


class _VisitedFaces:
    """Facce che hanno eseguito il blocco in almeno un salto (maschera su device, letta nel backward)."""

    def __init__(self) -> None:
        self.mask: Optional[torch.Tensor] = None
        self._host: Optional[List[bool]] = None

    def add(self, counts: torch.Tensor) -> None:
        hit = counts > 0
        self.mask = hit if self.mask is None else (self.mask | hit)

    def host(self, n_faces: int) -> List[bool]:
        if self._host is None:
            self._host = [False] * n_faces if self.mask is None else [bool(v) for v in self.mask.tolist()]
        return self._host


class _StackFaces(torch.autograd.Function):
    """torch.stack per tipo di parametro; nel backward le facce mai visitate ricevono None.

    Nel loop originale i parametri di una faccia non visitata non entrano nel
    grafo e restano con grad None (AdamW li salta: niente momento né weight
    decay); un grad di zeri cambierebbe l'aggiornamento. Per lo stesso motivo
    un tipo di parametro fuori dal grafo differenziabile (il router con routing
    hard) deve restare None: niente gradienti materializzati a zero.
    """

    @staticmethod
    def forward(ctx, visited: _VisitedFaces, n_faces: int, *flat: torch.Tensor):
        ctx.set_materialize_grads(False)
        ctx.visited = visited
        ctx.n_faces = n_faces
        return tuple(torch.stack(flat[k : k + n_faces]) for k in range(0, len(flat), n_faces))

    @staticmethod
    def backward(ctx, *grads):
        hit = ctx.visited.host(ctx.n_faces)
        out: List[Optional[torch.Tensor]] = [None, None]
        for g in grads:
            if g is None:
                out.extend([None] * ctx.n_faces)
            else:
                out.extend(gf if h else None for gf, h in zip(g.unbind(0), hit))
        return tuple(out)


def stack_bank(bank: nn.Module) -> Dict[str, object]:
    """Parametri della banca impilati per face id: ln (F,d), w1 (F,h,d), w2 (F,d,h), router (F,4,d).

    Con autograd attivo lo stack è nel grafo (i gradienti tornano ai parametri
    per faccia) e ``st["visited"]`` va aggiornato dal cammino. Senza gradienti
    il risultato resta in cache finché i parametri non cambiano (data_ptr e
    contatore di versione).
    """
    blocks = _face_blocks(bank)
    params = [_block_params(b) for b in blocks]
    n_faces = len(blocks)
    grad_mode = torch.is_grad_enabled() and any(p.requires_grad for ps in params for p in ps)
    if grad_mode:
        visited = _VisitedFaces()
        flat = [ps[i] for i in range(len(_NAMES)) for ps in params]
        st: Dict[str, object] = dict(zip(_NAMES, _StackFaces.apply(visited, n_faces, *flat)))
        st["visited"] = visited
        bank.__dict__.pop("_fast_walk_cache", None)
    else:
        key = (torch.is_inference_mode_enabled(),) + tuple(
            (p.data_ptr(), p._version, p.device, p.dtype) for ps in params for p in ps
        )
        cached = bank.__dict__.get("_fast_walk_cache")
        if cached is not None and cached[0] == key:
            return cached[1]
        st = {n: torch.stack([ps[i] for ps in params]) for i, n in enumerate(_NAMES)}
        st["visited"] = None
        bank.__dict__["_fast_walk_cache"] = (key, st)
    st["eps"] = float(blocks[0].norm.eps)
    st["num_faces"] = n_faces
    return st


def _triangle(
    xb: torch.Tensor, w: Sequence[torch.Tensor], eps: float, gate: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """TriangleBlock.forward a gruppi: xb (G, R, d), pesi (G, ...) di un blocco per gruppo.

    gate (G, R, 1), router v6: xb + gate · ff(norm(xb)), il peso non tocca il residuo.
    """
    ln_w, ln_b, w1, b1, w2, b2, wr, br = w
    d = xb.shape[-1]
    xn = F.layer_norm(xb, (d,), None, None, eps) * ln_w.unsqueeze(1) + ln_b.unsqueeze(1)
    hid = F.gelu(torch.baddbmm(b1.unsqueeze(1), xn, w1.transpose(1, 2)))
    delta = torch.baddbmm(b2.unsqueeze(1), hid, w2.transpose(1, 2))
    xo = xb + delta if gate is None else xb + gate * delta
    return xo, torch.baddbmm(br.unsqueeze(1), xo, wr.transpose(1, 2))


def _scatter_rows(x: torch.Tensor, dest: torch.Tensor, rows: int) -> torch.Tensor:
    """Buffer (rows, d) con x[i] alla riga dest[i]; dest == rows è la riga di scarto."""
    return x.new_zeros(rows + 1, x.shape[1]).index_copy(0, dest, x)[:rows]


def grouped_block_forward(
    st: Dict[str, object],
    x: torch.Tensor,
    face: torch.Tensor,
    active: torch.Tensor,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    dense_rows: int = DEFAULT_DENSE_ROWS,
    gate: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TriangleBlock della propria faccia su ogni token attivo, senza sincronizzazioni.

    x (B, d), face (B,) long, active (B,) bool, gate (B, 1) opzionale (router v6:
    peso straight-through sul solo ramo dell'esperto, disposto come x).
    Ritorna (x_out, logits, counts): (B, d), (B, 4) e token attivi per faccia (F,).
    Sulle righe inattive x_out/logits sono finiti ma arbitrari (il chiamante li
    scarta con torch.where).

    Nel loop originale ogni faccia fa i suoi Linear su tutte e sole le sue n_f
    righe, e il kernel dipende da n_f: su MPS uno per n_f <= 15 e uno per
    n_f >= 16, su CPU (BLAS) uno per n_f == 1 e uno per n_f >= 2; dentro lo
    stesso regime le righe sono bit-identiche qualunque sia n_f. Per restare
    bit-identici ci sono tre percorsi a forma fissa, ognuno nel regime giusto:
      - facce con n_f <= dense_rows (da 2 su CPU): un elemento di batch per
        faccia con dense_rows righe, sui pesi impilati senza gather;
      - su CPU, facce con un solo token: un elemento da 1 riga per faccia;
      - facce con n_f > dense_rows: blocchi da block_rows righe (padding solo
        nell'ultimo blocco di ogni faccia) con i pesi raccolti per blocco.
    """
    b, d = x.shape
    n_faces = int(st["num_faces"])
    s, c = int(block_rows), int(dense_rows)
    weights = [st[n] for n in _NAMES]  # (F, ...)
    eps = float(st["eps"])  # type: ignore[arg-type]
    dev = x.device
    single = dev.type == "cpu"
    lo = 2 if single else 1

    key = torch.where(active, face, torch.full_like(face, n_faces))  # inattivi → classe fittizia F
    face_c = key.clamp(max=n_faces - 1)
    onehot = (key.unsqueeze(1) == torch.arange(n_faces + 1, device=dev)).to(torch.int64)
    rank = onehot.cumsum(0).gather(1, key.unsqueeze(1)).squeeze(1) - 1  # posizione nella faccia
    counts = onehot[:, :n_faces].sum(0)
    n_tok = counts.gather(0, face_c)
    in_big = active & (n_tok > c)
    in_dense = active & (n_tok >= lo) & ~in_big
    zero = torch.zeros_like(rank)
    parts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []  # (maschera, x_out, logit, riga)

    def _gate(dest: torch.Tensor, rows: int, shape: Tuple[int, int]) -> Optional[torch.Tensor]:
        return None if gate is None else _scatter_rows(gate, dest, rows).view(*shape, 1)

    if c >= lo:
        dest = torch.where(in_dense, face * c + rank, torch.full_like(rank, n_faces * c))
        xo, lg = _triangle(
            _scatter_rows(x, dest, n_faces * c).view(n_faces, c, d), weights, eps, _gate(dest, n_faces * c, (n_faces, c))
        )
        parts.append((in_dense, xo.reshape(-1, d), lg.reshape(n_faces * c, -1), torch.where(in_dense, dest, zero)))
    if single:
        alone = active & (n_tok == 1)
        dest = torch.where(alone, face, torch.full_like(face, n_faces))
        xo, lg = _triangle(
            _scatter_rows(x, dest, n_faces).unsqueeze(1), weights, eps, _gate(dest, n_faces, (n_faces, 1))
        )
        parts.append((alone, xo.reshape(-1, d), lg.reshape(n_faces, -1), face_c))

    # Limite statico: sum_f ceil(n_f / s) <= ceil(B / s) + facce con n_f > c.
    n_blocks = -(-b // s) + min(n_faces, b // (c + 1))
    rows = n_blocks * s
    nblk = torch.div(torch.where(counts > c, counts, 0) + (s - 1), s, rounding_mode="floor")
    blk_end = nblk.cumsum(0)
    blk_start = blk_end - nblk
    dest = torch.where(in_big, blk_start.gather(0, face_c) * s + rank, torch.full_like(rank, rows))
    blk_face = (blk_end.unsqueeze(0) <= torch.arange(n_blocks, device=dev).unsqueeze(1)).sum(1)
    blk_face = blk_face.clamp(max=n_faces - 1)  # blocchi oltre il totale: righe a zero, uscite ignorate
    gathered = [w.index_select(0, blk_face) for w in weights]
    xo, lg = _triangle(_scatter_rows(x, dest, rows).view(n_blocks, s, d), gathered, eps, _gate(dest, rows, (n_blocks, s)))
    row = torch.where(in_big, dest, zero)
    out = xo.reshape(rows, d).index_select(0, row)
    logits = lg.reshape(rows, -1).index_select(0, row)

    for mask, p_out, p_lg, p_row in parts:
        m = mask.unsqueeze(-1)
        out = torch.where(m, p_out.index_select(0, p_row), out)
        logits = torch.where(m, p_lg.index_select(0, p_row), logits)
    return out, logits, counts


def _build_paths(face0: torch.Tensor, codes: Sequence[torch.Tensor]) -> List[List[int]]:
    start = face0.tolist()
    if not codes:
        return [[int(f)] for f in start]
    table = torch.stack(list(codes), dim=1).tolist()
    return [[int(f)] + [c for c in row if c != -2] for f, row in zip(start, table)]


def fast_walk_token(
    model: nn.Module,
    h: torch.Tensor,
    bank: nn.Module,
    to_sphere: nn.Linear,
    record_path: bool = False,
    block_rows: Optional[int] = None,
    dense_rows: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[List[List[int]]]]:
    """Stessa semantica di CascadeMeshModel.walk_token (loop per faccia), vettorizzata."""
    b = h.shape[0]
    face, face_oh = model.project_to_face(h, to_sphere)
    face0 = face
    max_hops = int(model.walk.max_hops)
    if b == 0 or max_hops <= 0:
        return h, (_build_paths(face0, []) if record_path else None)

    st = stack_bank(bank)
    rows = int(block_rows or getattr(model, "fast_walk_block_rows", 0) or DEFAULT_BLOCK_ROWS)
    dense = dense_rows if dense_rows is not None else getattr(model, "fast_walk_dense_rows", None)
    dense = DEFAULT_DENSE_ROWS if dense is None else int(dense)
    soft = not model._hard_routing()
    active = torch.ones(b, dtype=torch.bool, device=h.device)
    codes: List[torch.Tensor] = []

    for hop in range(max_hops):
        if hop > 0 and soft and not bool(active.any()):
            break
        out, lg, counts = grouped_block_forward(st, h, face, active, rows, dense)
        if st["visited"] is not None:
            st["visited"].add(counts)  # type: ignore[union-attr]
        act = active.unsqueeze(-1)
        if face_oh is not None:
            # ST: w≈1 in forward; backward → face_oh → to_sphere / router
            w = face_oh.gather(1, face.unsqueeze(1))
            h = torch.where(act, out * w, h)
        else:
            h = torch.where(act, out, h)
        logits = torch.where(act, lg, torch.zeros_like(lg))
        next_face, exit_mask, side_oh = model._route_step(face, logits)
        moved = active & ~exit_mask
        if side_oh is not None:
            scale = side_oh.sum(dim=-1, keepdim=True)
            h = torch.where(act, h * scale, h)
            routed_oh = model._scatter_side_oh(face, side_oh)
            if face_oh is None:
                face_oh = routed_oh
            else:
                face_oh = torch.where(moved.unsqueeze(-1), routed_oh, face_oh)
        face = torch.where(moved, next_face, face)
        just_exit = active & exit_mask
        active = active & ~exit_mask
        if record_path:
            codes.append(torch.where(moved, face, torch.where(just_exit, -1, -2)))

    return h, (_build_paths(face0, codes) if record_path else None)


def fast_walk_token_v6(
    model: nn.Module,
    h: torch.Tensor,
    bank: nn.Module,
    to_sphere: Optional[nn.Linear],
    entry_ids: Optional[torch.Tensor] = None,
    record_path: bool = False,
    block_rows: Optional[int] = None,
    dense_rows: Optional[int] = None,
    active: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[List[List[int]]]]:
    """Cammino del router v6 / hash, vettorizzato: semantica di ``CascadeMeshModel._walk_token_v6_loop``.

    Nessuna sincronizzazione nel salto: senza l'uscita anticipata su
    ``active.any()`` i salti dopo che tutti i token sono usciti sono no-op.
    ``active`` opzionale (B,): token inattivi (pad/unk stadi 2/3) saltano le FFN;
    in ``record_path`` la faccia d'ingresso è ``-1``.
    """
    b = h.shape[0]
    face, w = model._v6_entry(h, bank, to_sphere, entry_ids)
    max_hops = int(model.walk.max_hops)
    if active is None:
        active = torch.ones(b, dtype=torch.bool, device=h.device)
    else:
        active = active.to(device=h.device, dtype=torch.bool)
    face0 = torch.where(active, face, torch.full_like(face, -1))
    if b == 0 or max_hops <= 0:
        return h, (_build_paths(face0, []) if record_path else None)

    st = stack_bank(bank)
    rows = int(block_rows or getattr(model, "fast_walk_block_rows", 0) or DEFAULT_BLOCK_ROWS)
    dense = dense_rows if dense_rows is not None else getattr(model, "fast_walk_dense_rows", None)
    dense = DEFAULT_DENSE_ROWS if dense is None else int(dense)
    codes: List[torch.Tensor] = []

    for _ in range(max_hops):
        out, lg, counts = grouped_block_forward(st, h, face, active, rows, dense, gate=w)
        if st["visited"] is not None:
            st["visited"].add(counts)  # type: ignore[union-attr]
        act = active.unsqueeze(-1)
        h = torch.where(act, out, h)
        logits = torch.where(act, lg, torch.zeros_like(lg))
        next_face, exit_mask, w = model._v6_route(face, logits)
        moved = active & ~exit_mask
        face = torch.where(moved, next_face, face)
        just_exit = active & exit_mask
        active = active & ~exit_mask
        if record_path:
            codes.append(torch.where(moved, face, torch.where(just_exit, -1, -2)))

    return h, (_build_paths(face0, codes) if record_path else None)
