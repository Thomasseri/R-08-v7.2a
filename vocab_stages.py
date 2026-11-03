#!/usr/bin/env python3
"""Vocab a stadi — pipeline SEQUENZIALE a cascata (NON encode greedy parallelo).

Flusso utente (testo → V1 → V2 → V3):

  1. Vocab 1 (pezzi ~1200, goTommy2): spezzetta TUTTO il testo in token base.
  2. Quei token passano a Vocab 2 (parole): crea/modifica/aggrega pezzi → parole
     (stesso tipo di passaggio: sequenza in → sequenza aggregata out).
  3. L'uscita di V2 passa a Vocab 3 (unioni): unisce 2/3 parole se frequenti.

API encode:
  encode_v1(text, pezzi) → pezzi_ids
  pass_to_v2(pezzi_ids, bundle) → parole_ids (+ surfaces)
  pass_to_v3(parole_ids|surfaces, bundle) → unioni stream
  encode_cascade(text, bundle) = V1 → V2 → V3
  align_stages_to_pezzi: ancora causale + carry-forward post-ancora
    (pad solo pre-prima-ancora; dopo, ultima parola/unione resta viva su spazi/pezzi)

Build:
  vocab2 imparato da parole ricostruite sulle sequenze di pezzi (V1);
  vocab3 da unioni frequenti sull'uscita di V2.

Wiring train (proposta, NON avviato): 3 stadi in SERIE
  emb_v1(pezzi) → mesh1 → transform → emb_v2(parole) → mesh2 → … → emb_v3(unioni).
  Non 3 stream paralleli “prova unioni|parole|pezzi”. Serve --confirm-stages-train.

Deprecato: encode_stages greedy “unioni poi parole poi pezzi” (ora alias di cascade).
"""
from __future__ import annotations

import json
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Union

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "vocab_stages"
DEFAULT_PEZZI = Path("/Users/thomasseri/Desktop/columns/logs/goTommy2_vocab_by_id.txt")

SPECIALS = [
    "<|end|>",
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|unk|>",
    "<|sep|>",
]

# pezzi che rompono la parola in V2 (aggregazione pezzi → parola)
_WORD_BREAK_PEZZI = frozenset(
    {
        "<spazio>",
        " ",
        "\t",
        "\n",
        ".",
        ",",
        ";",
        ":",
        "!",
        "?",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        '"',
        "'",
        "«",
        "»",
        "-",
        "—",
        "/",
        "\\",
    }
)


@dataclass
class StageVocab:
    name: str
    id_to_token: list[str]
    token_to_id: dict[str, int]
    path: Optional[Path] = None
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.id_to_token)

    @property
    def unk_id(self) -> int:
        return int(self.token_to_id.get("<|unk|>", 4))

    @property
    def eos_id(self) -> int:
        return int(self.token_to_id.get("<|eos|>", 2))

    @property
    def pad_id(self) -> int:
        return int(self.token_to_id.get("<|pad|>", 3))


@dataclass
class StageBundle:
    """Tre vocab separati + percorsi (id-space distinti)."""

    pezzi: StageVocab
    parole: StageVocab
    unioni: StageVocab
    root: Path

    def sizes(self) -> dict[str, int]:
        return {
            "pezzi": len(self.pezzi),
            "parole": len(self.parole),
            "unioni": len(self.unioni),
        }


@dataclass
class V1Out:
    """Uscita Vocab 1: tutto il testo spezzettato in pezzi."""

    pezzi_ids: list[int]
    surfaces: list[str]


@dataclass
class V2Out:
    """Uscita Vocab 2: pezzi aggregati in parole (passaggio da V1)."""

    parole_ids: list[int]
    surfaces: list[str]
    # gruppi di pezzi_ids che hanno formato ciascuna parola
    pezzi_groups: list[list[int]] = field(default_factory=list)


@dataclass
class V3Out:
    """Uscita Vocab 3: unioni 2/3 parole sulle superfici di V2.

    `stream` è la sequenza a cascata: ogni item è
      ("unione", surface, uid) oppure ("parola", surface, wid)
    dopo merge greedy trigram→bigram. `unioni_ids` allinea a parole V2
    (uid se la parola apre un'unione, altrimenti -1).
    """

    unioni_ids: list[int]
    stream: list[tuple[str, str, int]]
    surfaces: list[str] = field(default_factory=list)


@dataclass
class StageEncoded:
    """Risultato cascade V1→V2→V3 (id space distinti)."""

    pezzi_ids: list[int]
    parole_ids: list[int]
    unioni_ids: list[int]
    # traccia: ("v1"|"v2"|"v3", surface, id_or_ids)
    spans: list[tuple[str, str, object]]
    v1: Optional[V1Out] = None
    v2: Optional[V2Out] = None
    v3: Optional[V3Out] = None


# ---------------------------------------------------------------------------
# Vocab I/O
# ---------------------------------------------------------------------------


def load_pezzi_vocab(path: Path | str | None = None) -> StageVocab:
    """Carica goTommy2 (~1251) come stadio pezzi (V1)."""
    p = Path(path) if path is not None else DEFAULT_PEZZI
    id_to_token: dict[int, str] = {}
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            gid = int(parts[3])
            id_to_token[gid] = parts[4]
    if not id_to_token:
        raise RuntimeError(f"pezzi vocab vuoto: {p}")
    n = max(id_to_token) + 1
    tokens = [id_to_token.get(i, f"<unk_{i}>") for i in range(n)]
    return StageVocab(
        name="pezzi",
        id_to_token=tokens,
        token_to_id={t: i for i, t in enumerate(tokens)},
        path=p,
        meta={"source": str(p), "note": "goTommy2-scale ~1200 (1251)"},
    )


def _with_specials(tokens: Iterable[str]) -> list[str]:
    seen = set(SPECIALS)
    out = list(SPECIALS)
    for t in tokens:
        if t in seen or not t:
            continue
        seen.add(t)
        out.append(t)
    return out


def make_stage_vocab(name: str, tokens: Iterable[str], *, meta: Optional[dict] = None) -> StageVocab:
    id_to_token = _with_specials(tokens)
    return StageVocab(
        name=name,
        id_to_token=id_to_token,
        token_to_id={t: i for i, t in enumerate(id_to_token)},
        meta=dict(meta or {}),
    )


def save_stage_vocab(vocab: StageVocab, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# stage={vocab.name} size={len(vocab)}\n")
        fh.write("# colonne: codice\\tscheda\\triga\\tid\\tpezzo\n")
        for i, tok in enumerate(vocab.id_to_token):
            scheda = 0 if i < 1000 else 1
            riga = i if i < 1000 else (i - 1000)
            code = f"{scheda:02d}{riga:03d}"
            fh.write(f"{code}\t{scheda:02d}\t{riga:03d}\t{i}\t{tok}\n")
    meta_path = path.with_suffix(".meta.json")
    blob = {
        "name": vocab.name,
        "size": len(vocab),
        "path": str(path),
        "meta": vocab.meta,
        "specials": SPECIALS,
    }
    meta_path.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
    vocab.path = path


def load_stage_vocab(path: Path, name: Optional[str] = None) -> StageVocab:
    id_to_token: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            id_to_token[int(parts[3])] = parts[4]
    n = max(id_to_token) + 1 if id_to_token else 0
    tokens = [id_to_token.get(i, f"<unk_{i}>") for i in range(n)]
    meta: dict = {}
    meta_path = path.with_suffix(".meta.json")
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")).get("meta", {})
        except json.JSONDecodeError:
            meta = {}
    return StageVocab(
        name=name or path.stem,
        id_to_token=tokens,
        token_to_id={t: i for i, t in enumerate(tokens)},
        path=path,
        meta=meta,
    )


def save_bundle(bundle: StageBundle, root: Optional[Path] = None) -> Path:
    out = Path(root) if root is not None else bundle.root
    out.mkdir(parents=True, exist_ok=True)
    pezzi_dst = out / "00_pezzi_by_id.txt"
    parole_path = out / "01_parole_by_id.txt"
    unioni_path = out / "02_unioni_by_id.txt"
    save_stage_vocab(bundle.pezzi, pezzi_dst)
    save_stage_vocab(bundle.parole, parole_path)
    save_stage_vocab(bundle.unioni, unioni_path)
    src = bundle.pezzi.meta.get("source") or str(DEFAULT_PEZZI)
    (out / "00_pezzi_SOURCE.txt").write_text(
        f"source={src}\nsize={len(bundle.pezzi)}\n", encoding="utf-8"
    )

    manifest = {
        "design": "multi_vocab_stages_cascade",
        "note": (
            "Tre id-space separati in SERIE: testo→V1 pezzi (~1200)→V2 parole→V3 unioni. "
            "NON encode greedy parallelo. ~1200 = scala goTommy2 (1251), non 1.2M."
        ),
        "stages": {
            "0_pezzi": {"path": str(pezzi_dst), "size": len(bundle.pezzi)},
            "1_parole": {"path": str(parole_path), "size": len(bundle.parole)},
            "2_unioni": {"path": str(unioni_path), "size": len(bundle.unioni)},
        },
        "encode": (
            "encode_cascade = encode_v1 → pass_to_v2 → pass_to_v3. "
            "Wiring train: 3 stadi in serie (emb/mesh per stadio), non stream paralleli."
        ),
        "api": {
            "encode_v1": "text → pezzi_ids",
            "pass_to_v2": "pezzi_ids → parole_ids (aggrega pezzi)",
            "pass_to_v3": "parole surfaces → unioni (2/3 se frequenti)",
            "encode_cascade": "encode_v1 → pass_to_v2 → pass_to_v3",
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    bundle.root = out
    return out


def load_bundle(root: Path | str) -> StageBundle:
    root = Path(root)
    pezzi = load_stage_vocab(root / "00_pezzi_by_id.txt", "pezzi")
    parole = load_stage_vocab(root / "01_parole_by_id.txt", "parole")
    unioni = load_stage_vocab(root / "02_unioni_by_id.txt", "unioni")
    return StageBundle(pezzi=pezzi, parole=parole, unioni=unioni, root=root)


# ---------------------------------------------------------------------------
# V1 — spezzetta tutto in pezzi
# ---------------------------------------------------------------------------


def encode_pezzi_fallback(text: str, pezzi: StageVocab) -> list[int]:
    """Encode grezzo a pezzi: match longest token goTommy2, altrimenti char/UNK.

    Case-sensitive prima (A≠a nel vocab pezzi); lower solo se manca il match.
    Whitespace distinto come gotommy: \\n→<riga>, \\t→<tab>, spazi→<spazio>.
    """
    s = text
    ids: list[int] = []
    i = 0
    max_len = max((len(t) for t in pezzi.id_to_token if not t.startswith("<")), default=1)
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            # Distingue whitespace: allineato a gotommy_encode / token speciali pezzi.
            if ch == "\n":
                ws_tok = "<riga>"
            elif ch == "\t":
                ws_tok = "<tab>"
            elif ch == "\r":
                ws_tok = "<cr>"
            elif ch == "\xa0":
                ws_tok = "<nbsp>"
            else:
                ws_tok = "<spazio>"
            tid = pezzi.token_to_id.get(ws_tok)
            if tid is not None:
                ids.append(tid)
            i += 1
            continue
        matched = False
        upper = min(len(s), i + max_len)
        for j in range(upper, i, -1):
            frag = s[i:j]
            # Lookup case-sensitive PRIMA; lower solo fallback (es. "Hello"→H non h).
            tid = pezzi.token_to_id.get(frag)
            if tid is None and frag.isalpha():
                tid = pezzi.token_to_id.get(frag.lower())
            if tid is not None:
                ids.append(tid)
                i = j
                matched = True
                break
        if matched:
            continue
        tid = pezzi.token_to_id.get(ch)
        if tid is None and ch.isalpha():
            tid = pezzi.token_to_id.get(ch.lower())
        if tid is None:
            tid = pezzi.unk_id
        ids.append(tid)
        i += 1
    return ids


def encode_v1(text: str, pezzi: StageVocab) -> V1Out:
    """Vocab 1: spezzetta TUTTO il testo in pezzi/token base (~1200).

    Preferisce gotommy_encode (tokenizer ufficiale goTommy2); fallback locale se
    gptommy non è importabile o fallisce.
    """
    ids: list[int] | None = None
    try:
        from gptommy.encode import encode as gotommy_encode

        ids = gotommy_encode(text, pezzi)
    except Exception:
        ids = None
    if ids is None:
        ids = encode_pezzi_fallback(text, pezzi)
    surfaces = [
        pezzi.id_to_token[i] if 0 <= i < len(pezzi.id_to_token) else "<|unk|>" for i in ids
    ]
    return V1Out(pezzi_ids=ids, surfaces=surfaces)


def _is_word_break_surface(surf: str) -> bool:
    if not surf or surf.startswith("<|"):
        # specials di controllo: non rompono da soli (trattati come pezzi “vivi”
        # solo se non in _WORD_BREAK); <|unk|> resta nel gruppo parola
        return surf in _WORD_BREAK_PEZZI or surf == "<spazio>"
    if surf in _WORD_BREAK_PEZZI:
        return True
    # singolo char di punteggiatura
    if len(surf) == 1 and not surf.isalnum() and not surf.isalpha():
        return True
    return False


def _surface_from_pezzo(surf: str) -> str:
    """Contributo testuale di un pezzo alla parola ricostruita (salta meta-token)."""
    if not surf or surf.startswith("<"):
        return ""
    return surf


# ---------------------------------------------------------------------------
# V2 — aggrega pezzi → parole
# ---------------------------------------------------------------------------


def reconstruct_words_from_pezzi(
    pezzi_ids: Sequence[int],
    pezzi: StageVocab,
) -> tuple[list[str], list[list[int]]]:
    """Da sequenza pezzi → liste di superfici parola + gruppi pezzi_ids.

    Split su spazio/punteggiatura; concatena le superfici dei pezzi nel gruppo.
    Usato sia in build (conta parole da V1) sia in pass_to_v2.
    """
    surfaces: list[str] = []
    groups: list[list[int]] = []
    buf_surfs: list[str] = []
    buf_ids: list[int] = []

    def _flush() -> None:
        nonlocal buf_surfs, buf_ids
        if not buf_ids:
            buf_surfs = []
            return
        word = "".join(buf_surfs).lower()
        if word:
            surfaces.append(word)
            groups.append(list(buf_ids))
        buf_surfs = []
        buf_ids = []

    for pid in pezzi_ids:
        surf = pezzi.id_to_token[pid] if 0 <= pid < len(pezzi.id_to_token) else "<|unk|>"
        if _is_word_break_surface(surf):
            _flush()
            continue
        piece = _surface_from_pezzo(surf)
        if piece:
            buf_surfs.append(piece)
            buf_ids.append(int(pid))
        else:
            # meta-token senza contributo testuale: chiudi la parola corrente
            _flush()
    _flush()
    return surfaces, groups


def pass_to_v2(pezzi_ids: Sequence[int], bundle: StageBundle) -> V2Out:
    """Vocab 2: prende i pezzi di V1, aggrega/modifica in parole (id space parole)."""
    word_surfs, groups = reconstruct_words_from_pezzi(pezzi_ids, bundle.pezzi)
    parole_ids: list[int] = []
    for w in word_surfs:
        wid = bundle.parole.token_to_id.get(w)
        if wid is None:
            wid = bundle.parole.unk_id
        parole_ids.append(wid)
    return V2Out(parole_ids=parole_ids, surfaces=word_surfs, pezzi_groups=groups)


# ---------------------------------------------------------------------------
# V3 — unioni 2/3 parole sull'uscita V2
# ---------------------------------------------------------------------------


def pass_to_v3(
    v2: Union[V2Out, Sequence[str], Sequence[int]],
    bundle: StageBundle,
) -> V3Out:
    """Vocab 3: sull'uscita di V2, unisce 2/3 parole se presenti in vocab unioni.

    Merge greedy sinistra→destra: trigram poi bigram; altrimenti passa la parola.
    """
    if isinstance(v2, V2Out):
        surfaces = list(v2.surfaces)
        parole_ids = list(v2.parole_ids)
    elif v2 and isinstance(v2[0], int):  # type: ignore[index]
        parole_ids = [int(x) for x in v2]  # type: ignore[arg-type]
        surfaces = [
            bundle.parole.id_to_token[i]
            if 0 <= i < len(bundle.parole.id_to_token)
            else "<|unk|>"
            for i in parole_ids
        ]
    else:
        surfaces = [str(x) for x in v2]  # type: ignore[arg-type]
        parole_ids = [
            bundle.parole.token_to_id.get(s, bundle.parole.unk_id) for s in surfaces
        ]

    unioni_ids = [-1] * len(surfaces)
    stream: list[tuple[str, str, int]] = []
    out_surfaces: list[str] = []
    i = 0
    while i < len(surfaces):
        if i + 2 < len(surfaces):
            tri = f"{surfaces[i]} {surfaces[i + 1]} {surfaces[i + 2]}"
            uid = bundle.unioni.token_to_id.get(tri)
            if uid is not None:
                unioni_ids[i] = uid
                stream.append(("unione", tri, uid))
                out_surfaces.append(tri)
                i += 3
                continue
        if i + 1 < len(surfaces):
            bi = f"{surfaces[i]} {surfaces[i + 1]}"
            uid = bundle.unioni.token_to_id.get(bi)
            if uid is not None:
                unioni_ids[i] = uid
                stream.append(("unione", bi, uid))
                out_surfaces.append(bi)
                i += 2
                continue
        wid = parole_ids[i] if i < len(parole_ids) else bundle.parole.unk_id
        stream.append(("parola", surfaces[i], wid))
        out_surfaces.append(surfaces[i])
        i += 1

    return V3Out(unioni_ids=unioni_ids, stream=stream, surfaces=out_surfaces)


# ---------------------------------------------------------------------------
# Cascade + compat
# ---------------------------------------------------------------------------


def encode_cascade(text: str, bundle: StageBundle) -> StageEncoded:
    """Pipeline ufficiale: testo → encode_v1 → pass_to_v2 → pass_to_v3."""
    v1 = encode_v1(text, bundle.pezzi)
    v2 = pass_to_v2(v1.pezzi_ids, bundle)
    v3 = pass_to_v3(v2, bundle)
    spans: list[tuple[str, str, object]] = []
    for s, pid in zip(v1.surfaces, v1.pezzi_ids):
        spans.append(("v1", s, pid))
    for s, wid, grp in zip(v2.surfaces, v2.parole_ids, v2.pezzi_groups):
        spans.append(("v2", s, (wid, grp)))
    for kind, surf, tid in v3.stream:
        spans.append(("v3", f"{kind}:{surf}", tid))
    return StageEncoded(
        pezzi_ids=v1.pezzi_ids,
        parole_ids=v2.parole_ids,
        unioni_ids=v3.unioni_ids,
        spans=spans,
        v1=v1,
        v2=v2,
        v3=v3,
    )


def _word_union_ids_from_v3(v3: V3Out, n_words: int) -> list[int]:
    """Id unione per parola V2: solo sull'ULTIMA parola di ogni unione (causale).

    Non propagare l'uid sulle parole iniziali/intermedie dell'unione: sarebbe
    lookahead (si rivelerebbe il merge prima che le parole successive siano
    complete). Parole senza unione → -1.
    """
    out = [-1] * n_words
    wi = 0
    for kind, surf, tid in v3.stream:
        if kind == "unione":
            n = max(1, len(surf.split()))
            last = wi + n - 1
            if 0 <= last < n_words:
                out[last] = int(tid)
            wi += n
        else:
            wi += 1
    return out


def _align_anchor_for_word(
    pezzi: Sequence[int],
    pezzi_vocab: StageVocab,
    first_t: int,
    last_t: int,
) -> int:
    """Ancora causale per una parola: spazio/break successivo, altrimenti ultimo pezzo.

    Mai il primo pezzo se la parola è multi-pezzo (anti-lookahead).
    """
    n = len(pezzi)
    anchor = last_t
    if last_t + 1 < n:
        nxt = (
            pezzi_vocab.id_to_token[pezzi[last_t + 1]]
            if 0 <= pezzi[last_t + 1] < len(pezzi_vocab.id_to_token)
            else "<|unk|>"
        )
        if _is_word_break_surface(nxt):
            anchor = last_t + 1
    # sicurezza: multi-pezzo non deve mai ancorare sul primo
    if first_t != last_t and anchor == first_t:
        anchor = last_t
    return anchor


def _carry_forward_after_anchor(ids: list[int], pad_id: int) -> None:
    """In-place: dopo un id non-pad, copia l'ultimo visto sui pad successivi.

    Non tocca il prefisso di soli pad (anti-lookahead pre-ancora resta valido).
    """
    last = pad_id
    for i, v in enumerate(ids):
        if v != pad_id:
            last = int(v)
        elif last != pad_id:
            ids[i] = last


def _is_word_terminator(surf: str) -> bool:
    """Pezzo che chiude la parola in V2: break oppure pezzo senza testo (<riga>, <tab>, <|unk|>, …)."""
    return _is_word_break_surface(surf) or not _surface_from_pezzo(surf)


def _prefix_union_ids(surfaces: Sequence[str], bundle: StageBundle) -> list[int]:
    """Per ogni parola j: ultima unione di ``pass_to_v3`` sulle sole parole 0..j (-1 se nessuna).

    Il merge greedy decide in i guardando le parole i+1 e i+2: le decisioni con
    i <= j-2 coincidono con quelle sull'intera sequenza, la coda del prefisso è
    al più il bigramma (j-1, j).
    """
    uni = bundle.unioni.token_to_id
    n = len(surfaces)
    starts: list[int] = []
    uids: list[int] = []
    i = 0
    while i < n:
        size, uid = 1, None
        if i + 2 < n:
            uid = uni.get(f"{surfaces[i]} {surfaces[i + 1]} {surfaces[i + 2]}")
            size = 3
        if uid is None and i + 1 < n:
            uid = uni.get(f"{surfaces[i]} {surfaces[i + 1]}")
            size = 2
        if uid is None:
            size = 1
        starts.append(i)
        uids.append(-1 if uid is None else int(uid))
        i += size

    out = [-1] * n
    k = 0
    last = -1
    for j in range(n):
        while k < len(starts) and starts[k] <= j - 2:
            if uids[k] >= 0:
                last = uids[k]
            k += 1
        cur = last
        if k < len(starts) and starts[k] == j - 1:
            uid = uni.get(f"{surfaces[j - 1]} {surfaces[j]}")
            if uid is not None:
                cur = int(uid)
        out[j] = cur
    return out


def _align_causal(enc: StageEncoded, bundle: StageBundle) -> tuple[list[int], list[int], list[int]]:
    """Allineamento in cui la feature in t dipende solo da pezzi[:t+1] (vedi align_stages_to_pezzi)."""
    pezzi = list(enc.pezzi_ids)
    n = len(pezzi)
    pad_w = bundle.parole.pad_id
    pad_u = bundle.unioni.pad_id
    parole_al = [pad_w] * n
    unioni_al = [pad_u] * n
    if n == 0:
        return pezzi, parole_al, unioni_al
    pv = bundle.pezzi
    word_last: list[int] = []
    in_word = False
    for t, pid in enumerate(pezzi):
        surf = pv.id_to_token[pid] if 0 <= pid < len(pv.id_to_token) else "<|unk|>"
        if _is_word_terminator(surf):
            in_word = False
            continue
        if not in_word:
            word_last.append(t)
            in_word = True
        word_last[-1] = t
    if enc.v2 is not None:
        surfaces = list(enc.v2.surfaces)
    else:
        surfaces, _ = reconstruct_words_from_pezzi(pezzi, pv)
    if len(surfaces) != len(word_last) or len(enc.parole_ids) != len(word_last):
        raise ValueError("align causale: parole V2 e gruppi di pezzi non coincidono")
    word_uids = _prefix_union_ids(surfaces, bundle)
    for w, last_t in enumerate(word_last):
        anchor = last_t + 1
        if anchor >= n:
            continue
        parole_al[anchor] = int(enc.parole_ids[w])
        if word_uids[w] >= 0:
            unioni_al[anchor] = word_uids[w]
    _carry_forward_after_anchor(parole_al, pad_w)
    _carry_forward_after_anchor(unioni_al, pad_u)
    return pezzi, parole_al, unioni_al


def align_stages_to_pezzi(
    enc: StageEncoded,
    bundle: StageBundle,
    causal: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    """Allinea parole/unioni alla timeline pezzi (stessa lunghezza).

    ``causal=True`` (``--causal-align``): la feature in t è quella calcolabile
    dai soli pezzi[:t+1]. Rispetto alla policy qui sotto cambia in tre punti:
      - l'ancora di una parola è sempre il pezzo che la chiude in V2 (break
        oppure pezzo senza testo: <riga>, <tab>, <cr>, <nbsp>, <|unk|>, …),
        mai il suo ultimo pezzo: finché non arriva il terminatore la parola
        potrebbe continuare;
      - una parola non chiusa a fine sequenza non viene mai rivelata;
      - l'unione mostrata sull'ancora della parola j è l'ultima unione del
        merge V3 sulle sole parole 0..j (il greedy trigramma→bigramma
        sull'intera sequenza guarda due parole avanti).

    Policy (due pass):

    1) Ancora CAUSALE (anti-lookahead; coerente con gen greedy che mette
       pad_parole/pad_unioni sui pezzi nuovi finché la parola non è chiusa):

       - Non assegnare id parola/unione al PRIMO pezzo di una parola multi-pezzo
         (leak: rivelerebbe l'identità prima di aver visto tutti i pezzi).
       - Id parola: solo sull'ultimo pezzo della parola, oppure sullo
         spazio/break immediatamente successivo se presente.
       - Id unione: solo sull'ancora dell'ultima parola coperta dall'unione
         (vedi `_word_union_ids_from_v3`).
       - Pezzi intermedi / primo pezzo (multi-pezzo) / break non-ancora →
         ``pad`` di parole/unioni. Non si usa ``unk`` qui: unk = superficie
         assente dal vocab V2/V3, non “non ancora nota”.
       - Parola mono-pezzo senza break successivo: ancora = quel pezzo
         (superficie già completa → non è lookahead).

    2) Carry-forward post-ancora: una volta vista un'ancora, l'ultima
       parola/unione nota si propaga alle posizioni successive ancora in
       ``pad`` (spazi, pezzi della parola successiva, punteggiatura extra),
       finché non arriva una nuova ancora. Così su ``"Answer: "`` lo stadio
       V2 vede ``answer`` anche sullo ``<spazio>`` finale, non un reset a pad.
       Il prefisso pre-prima-ancora resta pad (niente leak sul primo pezzo).
    """
    if causal:
        return _align_causal(enc, bundle)
    pezzi = list(enc.pezzi_ids)
    n = len(pezzi)
    pad_w = bundle.parole.pad_id
    pad_u = bundle.unioni.pad_id
    parole_al = [pad_w] * n
    unioni_al = [pad_u] * n
    if n == 0:
        return pezzi, parole_al, unioni_al

    pezzi_vocab = bundle.pezzi
    n_words = len(enc.parole_ids)
    word_first = [-1] * n_words
    word_last = [-1] * n_words
    word_idx = -1
    in_word = False
    for t, pid in enumerate(pezzi):
        surf = (
            pezzi_vocab.id_to_token[pid]
            if 0 <= pid < len(pezzi_vocab.id_to_token)
            else "<|unk|>"
        )
        if _is_word_break_surface(surf):
            in_word = False
            continue
        piece = _surface_from_pezzo(surf)
        if piece:
            if not in_word:
                word_idx += 1
                in_word = True
                if 0 <= word_idx < n_words:
                    word_first[word_idx] = t
            if 0 <= word_idx < n_words:
                word_last[word_idx] = t
        else:
            in_word = False

    if enc.v3 is not None:
        word_uids = _word_union_ids_from_v3(enc.v3, n_words)
    else:
        word_uids = [-1] * n_words
        # fallback senza stream: unioni_ids è start-aligned in V3Out;
        # senza n-gram non possiamo spostare sull'ultima parola.
        for i, uid in enumerate(enc.unioni_ids):
            if 0 <= i < n_words and int(uid) >= 0:
                word_uids[i] = int(uid)

    for w in range(n_words):
        first_t = word_first[w]
        last_t = word_last[w]
        if first_t < 0 or last_t < 0:
            continue
        anchor = _align_anchor_for_word(pezzi, pezzi_vocab, first_t, last_t)
        parole_al[anchor] = int(enc.parole_ids[w])
        uid = word_uids[w] if w < len(word_uids) else -1
        if uid >= 0:
            unioni_al[anchor] = int(uid)

    _carry_forward_after_anchor(parole_al, pad_w)
    _carry_forward_after_anchor(unioni_al, pad_u)
    return pezzi, parole_al, unioni_al


def encode_stages(text: str, bundle: StageBundle) -> StageEncoded:
    """DEPRECATED: ex greedy parallelo; ora alias di encode_cascade (V1→V2→V3)."""
    warnings.warn(
        "encode_stages greedy è deprecato; usa encode_cascade / encode_v1→pass_to_v2→pass_to_v3",
        DeprecationWarning,
        stacklevel=2,
    )
    return encode_cascade(text, bundle)


# ---------------------------------------------------------------------------
# Build: vocab2 da pezzi (V1), vocab3 da uscita V2
# ---------------------------------------------------------------------------


def iter_v1_word_surfaces(
    texts: Iterator[str],
    pezzi: StageVocab,
    *,
    max_chars: int = 0,
) -> Iterator[list[str]]:
    """Per ogni testo: V1 → parole ricostruite da pezzi (senza ancora Vocab 2)."""
    seen = 0
    for text in texts:
        if max_chars > 0 and seen >= max_chars:
            break
        seen += len(text)
        v1 = encode_v1(text, pezzi)
        surfs, _ = reconstruct_words_from_pezzi(v1.pezzi_ids, pezzi)
        if surfs:
            yield surfs


def count_from_cascade(
    texts: Iterator[str],
    pezzi: StageVocab,
    *,
    max_chars: int = 0,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Conta parole (da V1→ricostruzione) e bi/tri sulla stessa sequenza parole.

    Nota build: vocab3 si impara dalle unioni sulle superfici che V2 vedrà
    (parole ricostruite da pezzi). Dopo aver creato Vocab 2 si può rivalidare
    con pass_to_v2; le superfici coincidono.
    """
    uni: Counter[str] = Counter()
    bi: Counter[str] = Counter()
    tri: Counter[str] = Counter()
    for surfs in iter_v1_word_surfaces(texts, pezzi, max_chars=max_chars):
        uni.update(surfs)
        if len(surfs) >= 2:
            bi.update(f"{surfs[i]} {surfs[i + 1]}" for i in range(len(surfs) - 1))
        if len(surfs) >= 3:
            tri.update(
                f"{surfs[i]} {surfs[i + 1]} {surfs[i + 2]}" for i in range(len(surfs) - 2)
            )
    return dict(uni), dict(bi), dict(tri)


def count_ngrams_from_texts(
    texts: Iterator[str],
    *,
    max_chars: int = 0,
    pezzi: StageVocab | None = None,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Build helper: conta via cascade V1→parole ricostruite (non word-tokenize grezzo)."""
    if pezzi is None:
        pezzi = load_pezzi_vocab()
    return count_from_cascade(texts, pezzi, max_chars=max_chars)


def build_stages_from_counts(
    word_counts: dict[str, int],
    bigram_counts: dict[str, int],
    trigram_counts: dict[str, int],
    *,
    pezzi_path: Path | str | None = None,
    min_word_freq: int = 3,
    min_union_freq: int = 5,
    max_words: int = 50000,
    max_unions: int = 20000,
    out_dir: Path | str | None = None,
) -> StageBundle:
    """Costruisce V1 (carica pezzi), V2 da conteggi parole-da-pezzi, V3 da unioni su V2."""
    pezzi = load_pezzi_vocab(pezzi_path)
    ranked_words = sorted(
        ((w, c) for w, c in word_counts.items() if c >= min_word_freq and len(w) > 1),
        key=lambda x: (-x[1], x[0]),
    )[:max_words]
    parole = make_stage_vocab(
        "parole",
        (w for w, _ in ranked_words),
        meta={
            "min_word_freq": min_word_freq,
            "n_candidates": len(ranked_words),
            "source": "reconstructed_from_v1_pezzi",
            "top": ranked_words[:20],
        },
    )

    union_items: list[tuple[str, int, str]] = []
    for t, c in trigram_counts.items():
        if c >= min_union_freq:
            union_items.append((t, c, "tri"))
    for t, c in bigram_counts.items():
        if c >= min_union_freq:
            union_items.append((t, c, "bi"))
    union_items.sort(key=lambda x: (-x[1], 0 if x[2] == "tri" else 1, x[0]))
    union_items = union_items[:max_unions]
    unioni = make_stage_vocab(
        "unioni",
        (t for t, _, _ in union_items),
        meta={
            "min_union_freq": min_union_freq,
            "n_candidates": len(union_items),
            "source": "frequent_unions_on_v2_surfaces",
            "n_trigram": sum(1 for _, _, k in union_items if k == "tri"),
            "n_bigram": sum(1 for _, _, k in union_items if k == "bi"),
            "top": [(t, c, k) for t, c, k in union_items[:20]],
        },
    )

    root = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    bundle = StageBundle(pezzi=pezzi, parole=parole, unioni=unioni, root=root)
    save_bundle(bundle, root)
    counts_path = root / "counts_summary.json"
    counts_path.write_text(
        json.dumps(
            {
                "pipeline": "v1_pezzi → v2_parole(from pezzi) → v3_unioni(from v2)",
                "n_word_types": len(word_counts),
                "n_bigram_types": len(bigram_counts),
                "n_trigram_types": len(trigram_counts),
                "parole_kept": len(parole) - len(SPECIALS),
                "unioni_kept": len(unioni) - len(SPECIALS),
                "min_word_freq": min_word_freq,
                "min_union_freq": min_union_freq,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return bundle
