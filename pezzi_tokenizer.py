#!/usr/bin/env python3
"""Tokenizer pezzi goTommy2 locale: sostituisce ``gptommy.encode`` senza bytecode in /tmp.

``r08/gptommy/encode.py`` è un wrapper che carica il bytecode originale da
``/tmp/encode_ORIGINAL.cpython-314.pyc`` (cancellato dalla pulizia di macOS) o da
``__pycache__`` solo se >1000 byte; lì c'è ormai il wrapper stesso ricompilato
(722 byte), quindi l'import fallisce. Il sorgente originale non esiste più.

Il contratto dell'encoder sopravvive nel docstring del bytecode originale di
``pipeline.cpython-314.pyc`` (20 set): copia in
``vendor/gptommy_pycache_backup_20261019/``, da leggere con marshal senza
importare ``gptommy.pipeline`` (l'import sovrascriverebbe l'originale in r08):

  1. NFC  2. speciali letterali ``<|...|>``  3. whitespace (spazio→<spazio>,
  \\n→<riga>, \\t→<tab>, \\r→<cr>, NBSP→<nbsp>; pezzi indent se esistono)
  4. cifre: coppie 00–99 se in vocab, altrimenti cifra singola
  5. longest-match greedy intra-parola, fallback 1 carattere, poi <|unk|>
  6. case: match sul testo così com'è, poi ``lower()`` se non c'è pezzo.

Col vocab pezzi 1251 (nessuna coppia di cifre, nessun pezzo con spazi o indent,
nessun pezzo multi-carattere con maiuscole) il contratto coincide con
``vocab_stages.encode_pezzi_fallback``. ``encode`` replica esattamente quella
lettura (``TRAINING_VARIANT``); le altre letture ammesse dal testo del contratto
restano disponibili via ``encode_variant`` solo per la verifica
(``verify_tokenizer_equivalence.py``).
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
GOTTOMMY2_VOCAB = Path("/Users/thomasseri/Desktop/columns/logs/goTommy2_vocab_by_id.txt")
LOCAL_PEZZI_VOCAB = ROOT / "vocab_stages" / "00_pezzi_by_id.txt"

WS_PIECES = {"\n": "<riga>", "\t": "<tab>", "\r": "<cr>", "\xa0": "<nbsp>", " ": "<spazio>"}
DECODE_WS = {"<spazio>": " ", "<riga>": "\n", "<tab>": "\t"}

TRAINING_VARIANT = {
    "nfc": False,
    "literal_specials": False,
    "case_rule": "per_length",
    "unicode_ws": "isspace",
}


@dataclass
class Vocab:
    id_to_token: List[str]
    token_to_id: Dict[str, int]
    path: Optional[Path] = None


def load_vocab(path: Path | str | None = None) -> Vocab:
    """Stesso formato TSV di ``gptommy.vocab.load_vocab`` (codice, scheda, riga, id, pezzo)."""
    p = Path(path) if path is not None else (
        LOCAL_PEZZI_VOCAB if LOCAL_PEZZI_VOCAB.is_file() else GOTTOMMY2_VOCAB
    )
    id_to_token: Dict[int, str] = {}
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            id_to_token[int(parts[3])] = parts[4]
    if not id_to_token:
        raise RuntimeError(f"vocab vuoto: {p}")
    n = max(id_to_token) + 1
    tokens = [id_to_token.get(i, f"<unk_{i}>") for i in range(n)]
    return Vocab(id_to_token=tokens, token_to_id={t: i for i, t in enumerate(tokens)}, path=p)


class _Tables:
    __slots__ = ("max_len", "specials", "unk_id")

    def __init__(self, vocab) -> None:
        toks = vocab.id_to_token
        self.max_len = max((len(t) for t in toks if not t.startswith("<")), default=1)
        self.specials = sorted(
            (t for t in toks if t.startswith("<") and len(t) > 1 and t.endswith(">")),
            key=len,
            reverse=True,
        )
        self.unk_id = int(vocab.token_to_id.get("<|unk|>", 4))


_TABLE_CACHE: Dict[int, tuple] = {}


def _tables(vocab) -> _Tables:
    key = id(vocab.token_to_id)
    hit = _TABLE_CACHE.get(key)
    if hit is not None and hit[0] is vocab.token_to_id:
        return hit[1]
    tab = _Tables(vocab)
    _TABLE_CACHE[key] = (vocab.token_to_id, tab)
    return tab


def _is_ws(ch: str, unicode_ws: str) -> bool:
    if unicode_ws == "isspace":
        return ch.isspace()
    return ch in WS_PIECES


def encode_variant(
    text: str,
    vocab,
    *,
    nfc: bool = False,
    literal_specials: bool = False,
    case_rule: str = "per_length",
    unicode_ws: str = "isspace",
) -> List[int]:
    """Encoder del contratto gpTommy con le letture alternative esplicite.

    case_rule:
      ``per_length``   a ogni lunghezza prova il frammento così com'è, poi lower()
                       se alfabetico (= encode_pezzi_fallback; "Questo" → questo)
      ``per_position`` lower() solo se nessun pezzo matcha così com'è a quella
                       posizione ("Questo" → Q u e s t o, visto che A–Z esistono)
    unicode_ws:
      ``isspace``      ogni carattere str.isspace() è whitespace (default <spazio>)
      ``listed``       solo spazio, \\n, \\t, \\r, NBSP; gli altri vanno al longest-match
    literal_specials: speciali ``<...>`` di qualunque lunghezza consumati per primi
                      (il longest-match generico arriva solo a max_len=8 caratteri).
    """
    if case_rule not in ("per_length", "per_position"):
        raise ValueError(f"case_rule sconosciuta: {case_rule}")
    s = unicodedata.normalize("NFC", text) if nfc else text
    t2i = vocab.token_to_id
    tab = _tables(vocab)
    ids: List[int] = []
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if _is_ws(ch, unicode_ws):
            tid = t2i.get(WS_PIECES.get(ch, "<spazio>"))
            if tid is not None:
                ids.append(tid)
            i += 1
            continue
        if literal_specials and ch == "<":
            hit = next((sp for sp in tab.specials if s.startswith(sp, i)), None)
            if hit is not None:
                ids.append(t2i[hit])
                i += len(hit)
                continue
        upper = min(n, i + tab.max_len)
        tid = None
        j = upper
        while j > i:
            frag = s[i:j]
            tid = t2i.get(frag)
            if tid is None and case_rule == "per_length" and frag.isalpha():
                tid = t2i.get(frag.lower())
            if tid is not None:
                break
            j -= 1
        if tid is None and case_rule == "per_position":
            j = upper
            while j > i:
                frag = s[i:j]
                if frag.isalpha():
                    tid = t2i.get(frag.lower())
                    if tid is not None:
                        break
                j -= 1
        if tid is not None:
            ids.append(tid)
            i = j
            continue
        tid = t2i.get(ch)
        if tid is None and ch.isalpha():
            tid = t2i.get(ch.lower())
        ids.append(tab.unk_id if tid is None else tid)
        i += 1
    return ids


def encode(text: str, vocab) -> List[int]:
    """Encode pezzi usato dal training (stessa lettura di encode_pezzi_fallback)."""
    return encode_variant(text, vocab, **TRAINING_VARIANT)


def decode(ids: Sequence[int], vocab) -> str:
    """Concatena i pezzi; sostituisce solo <spazio>/<riga>/<tab> (contratto gpTommy)."""
    toks = vocab.id_to_token
    out: List[str] = []
    for tid in ids:
        tok = toks[tid] if 0 <= int(tid) < len(toks) else "<|unk|>"
        out.append(DECODE_WS.get(tok, tok))
    return "".join(out)


def cascade_from_pezzi(pezzi_ids: Sequence[int], bundle):
    """StageEncoded da pezzi già codificati: pass_to_v2 → pass_to_v3 (come encode_cascade)."""
    from vocab_stages import StageEncoded, pass_to_v2, pass_to_v3

    ids = [int(x) for x in pezzi_ids]
    v2 = pass_to_v2(ids, bundle)
    v3 = pass_to_v3(v2, bundle)
    return StageEncoded(
        pezzi_ids=ids,
        parole_ids=list(v2.parole_ids),
        unioni_ids=list(v3.unioni_ids),
        spans=[],
        v2=v2,
        v3=v3,
    )


def encode_cascade(text: str, bundle):
    """vocab_stages.encode_cascade con V1 = ``encode`` locale (mai gptommy)."""
    return cascade_from_pezzi(encode(text, bundle.pezzi), bundle)


def encode_stages_aligned(text: str, bundle) -> Tuple[List[int], List[int], List[int]]:
    """(pezzi, parole, unioni) allineati sulla timeline pezzi, come nel training."""
    from vocab_stages import align_stages_to_pezzi

    return align_stages_to_pezzi(encode_cascade(text, bundle), bundle)
