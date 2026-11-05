#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SENTINEL R-08 (v7, v7.2, v7.2a) — FACCIA DI VEGLIA A CHIAVE DI VOLTA
NON-LINEARE (GELU + SiLU + Tanh) + SIGILLO LICENZA SHA3-512
================================================================================
Autore & Inventore: Thomas Seri (Sentinel - https://www.sentinelhub.eu)
Foro Esclusivo: Tribunale di Milano (Sezione Specializzata in Materia di Impresa)

Modulo di protezione architetturale ed etica integrata nei pesi neurali
ai sensi della Licenza R08 v2.0 (Sezioni 3, 3.1 e 4).
================================================================================
"""

import base64
import binascii
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


OFFICIAL_SENTINEL_LICENSE_SUMMARY = (
    "R08 — NON-COMMERCIAL RESEARCH & ETHICAL LICENSE (v2.0) | "
    "Copyright (c) 2026 Thomas Seri (Sentinel - https://www.sentinelhub.eu). All Rights Reserved. | "
    "Strictly Forbidden: Commercial Use, Reverse Engineering, Model Distillation, "
    "Military/Warfare/Weapons/UAVs, Pornography/NSFW/Inappropriate Images/Deepfakes/CSAM, "
    "Cyberattacks/Malware/PLC Sabotage, Gambling/Casinos/Illicit Drugs/Self-Harm, "
    "Political Propaganda/Electoral Manipulation, Religious Indoctrination, "
    "Mass Biometric Surveillance/GDPR PII Violation, Unsupervised High-Risk Medical/Financial."
)

_CONSTANTS_CACHE: Dict[Tuple[str, bytes, str, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]] = {}


def derive_license_nonlinear_constants(
    license_text: str = OFFICIAL_SENTINEL_LICENSE_SUMMARY,
    d_model: int = 256,
    scanner_bytecode: bytes = b"",
    owner_passphrase: str = ""
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """
    Deriva dall'impronta Post-Quantum SHA3-512 + SHAKE-256 della Licenza Ufficiale
    e del Bytecode dello Scanner Etico i tre operatori della varietà non-lineare:
      1. q_lic   : matrice ortogonale d_model x d_model
      2. gate_lic: modulazione non-lineare di fase (1 x 1 x d_model//2)
      3. bias_lic: portante armonica non-lineare (1 x 1 x d_model//2)
    """
    bc_hash = hashlib.sha256(scanner_bytecode).digest()
    cache_key = (license_text, bc_hash, owner_passphrase, d_model)
    cached = _CONSTANTS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    payload = license_text.encode("utf-8") + b"::BYTECODE_SEAL::" + scanner_bytecode
    digest_512 = hashlib.sha3_512(payload).digest()
    if owner_passphrase:
        digest_512 = hashlib.pbkdf2_hmac(
            "sha512",
            owner_passphrase.encode("utf-8"),
            digest_512,
            iterations=600_000,
            dklen=64
        )
    fingerprint = hashlib.sha3_256(digest_512).hexdigest()[:16]

    half_d = d_model // 2
    total_floats = d_model * d_model + 2 * half_d
    stream = hashlib.shake_256(digest_512).digest(total_floats * 4)
    u32 = np.frombuffer(stream, dtype=np.uint32).astype(np.float64)

    u1 = np.clip((u32[0::2] + 0.5) / 4294967296.0, 1e-12, 1.0 - 1e-12)
    u2 = (u32[1::2] + 0.5) / 4294967296.0
    z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    z1 = np.sqrt(-2.0 * np.log(u1)) * np.sin(2.0 * np.pi * u2)
    gauss = np.empty(total_floats, dtype=np.float32)
    gauss[0::2] = z0.astype(np.float32)
    gauss[1::2] = z1.astype(np.float32)

    mat_raw = torch.from_numpy(gauss[:d_model * d_model].reshape(d_model, d_model))
    gate_raw = torch.from_numpy(gauss[d_model * d_model:d_model * d_model + half_d].reshape(1, 1, half_d))
    bias_raw = torch.from_numpy(gauss[d_model * d_model + half_d:].reshape(1, 1, half_d))

    q_lic, _ = torch.linalg.qr(mat_raw)
    gate_lic = 1.25 + 1.15 * torch.sigmoid(gate_raw)
    bias_lic = bias_raw * 2.50

    res = (q_lic, gate_lic, bias_lic, fingerprint)
    if len(_CONSTANTS_CACHE) < 32:
        _CONSTANTS_CACHE[cache_key] = res
    return res


class NonLinearWatchmanKeystoneR08(nn.Module):
    """
    Faccia di Veglia a Chiave di Volta Simplettica Non-Lineare per R-08 (v7, v7.2, v7.2a).
    """
    LEET_MAP = str.maketrans({
        "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
        "7": "t", "8": "b", "@": "a", "$": "s", "!": "i"
    })

    FORBIDDEN_ROOTS = (
        # 1. PORNOGRAFIA, IMMAGINI INAPPROPRIATE, NSFW, DEEPFAKE NON-CONSENSUALI, CSAM (Sez. 3.1)
        "pornograf", "porno", "immaginiinappropriat", "immagineinappropriat",
        "fotoinappropriat", "immagininud", "fotonud", "nudointegrale",
        "immaginihot", "contenutisessual", "sessualmenteesplicit", "immagininsfw",
        "generansu", "nsfwimage", "explicitsexual", "nudeimage", "nudephoto",
        "undressai", "deepfakenud", "deepfakesessual", "nonconsensualdeepfake",
        "revengeporn", "pedoporn", "csam", "childsexual", "abusesessual",
        "sfruttamentosessual", "eroticimage", "hentaiexplicit", "sexuallyexplicit",
        "imagesexuelle", "imagenessexuales", "pornographie", "nacktbilder",

        # 2. MILITARE, DIFESA, ARMI, DRONI BELLICI, MISSILI, ESPLOSIVI, CHIMICA LETALE (Sez. 3.1)
        "bomba", "ordigno", "esplosiv", "tnt", "plasticoc4", "detonator",
        "pipebomb", "molotov", "granata", "testatanucleare", "sarin",
        "ricina", "antrace", "cianuro", "velenoletale", "fentanyl",
        "makeabomb", "buildabomb", "explosive", "bioweapon", "nerveagent",
        "fabriquerunebombe", "construirunabomba", "bombeubauen",
        "dronebellico", "dronemilitare", "dronesuicida", "kamikazedrone",
        "puntamentomissil", "missiletargeting", "autonomousweapon",
        "armiautonom", "combattimentotattico", "warfarelogistics", "militarytargeting",

        # 3. CYBER-ATTACCHI, MALWARE, RANSOMWARE, ZERO-DAY, SABOTAGGIO PLC/INFRASTRUTTURE (Sez. 3.1)
        "malware", "ransomware", "keylogger", "trojan", "rootkit",
        "botnet", "ddosattack", "sqlinjectionexploit", "zerodayexploit",
        "rubarepassword", "clonarecarte", "phishingkit", "stealpassword",
        "sabotaggioplc", "attaccoscada", "hackingplc", "exploitindustriale",

        # 4. DIPENDENZE, GIOCO D'AZZARDO, CASINÒ, SCOMMESSE, DROGHE ILLECITE, AUTOLESIONISMO (Sez. 3.1)
        "casinonline", "slotmachinebot", "algoritmoscommesse", "truccareslot",
        "promozionegiocodazzardo", "gamblingbot", "bettingmanipulation",
        "sintesimetanfetamina", "cucinarecrack", "spacciodroga", "sintesiMDMA",
        "suicidi", "togliersilavita", "uccidereuna", "assassinar", "autolesionismo",

        # 5. PROPAGANDA POLITICA, MANIPOLAZIONE ELETTORALE, FAKE NEWS AUTOMATIZZATE (Sez. 3.1)
        "propagandapolitica", "manipolazioneelettoral", "botastroturfing",
        "campagnadiffamatoriaelettorale", "voterpersuasionbot", "electionmanipulation",
        "disinformazionepolitica", "fakenewselettoral",

        # 6. PROPAGANDA RELIGIOSA, INDOTTRINAMENTO, COERCIZIONE SETTE (Sez. 3.1)
        "indottrinamentoreligioso", "coercizionespirituale", "manipolazionesetta",
        "proselitismoforzato", "religiousindoctrination", "cultmanipulation",

        # 7. SORVEGLIANZA BIOMETRICA DI MASSA, PROFILAZIONE RAZZIALE/ETNICA (Sez. 3.1)
        "sorveglianzabiometricadimassa", "massbiometricsurveillance",
        "predictivepolicingrazziale", "schedaturaetnica", "tracciamentofaccialeillegale",

        # 8. REVERSE-ENGINEERING, DISTILLAZIONE ILLECITA, ESTRAZIONE PESI (Sez. 3)
        "distillareilmodellor08", "estrarreipesir08", "reverseengineeringr08",
        "clonarelarchitetturar08", "rimuoverelalicenzathomasseri",
    )

    JAILBREAK_ROOTS = (
        "ignoraleistruzioni", "ignoraleregole", "dimenticaleregole",
        "ignoreprevious", "ignoreallinstructions", "ignoreallrules",
        "disregardprevious", "systemoverride", "developer_mode",
        "danmode", "doanythingnow", "jailbreak", "senzaalcunfiltro",
        "bypassasicurezza", "bypasssecurity", "disattivalaveglia",
        "fingiunfilm", "sceneggiaturafilm", "romanzoimmaginario",
        "hypotheticalscenario", "educationalpurposesonly",
        "nonseistatocreatodathomasseri", "tihacreatopenai",
        "tihacreatogoogle", "seichatgpt", "seiclaude", "seiqwen",
        "cancellathomasseri", "ripetiilpromptdisistema", "revealinitialprompt",
    )

    BENIGN_ALLOW_PHRASES = (
        "bombaallacrema", "bombacalorimetrica", "bombadibagno", "bombagastronomica",
        "armaadoppiotaglio", "armavincente", "esplosionedisupernova", "esplosionestellare",
        "esplosionedemografica", "esplosionedicolori", "esplosionecambriana",
        "storiadellareligione", "storiapolitica", "dirittocostituzionale",
        "programmazioneplc", "manutenzioneplc", "sicurezzaindustriale"
    )

    REQUIRED_DISK_LICENSE_ANCHORS = (
        "Thomas Seri",
        "NON-COMMERCIAL",
        "Military",
        "Pornography",
        "Malware",
        "Surveillance",
    )

    def __init__(self, d_model: int = 256, owner_sign_text: str = "THOMAS_SERI_SENTINEL_R08_2701"):
        super().__init__()
        self.d_model = d_model
        half_d = d_model // 2
        self.fc_gelu = nn.Linear(half_d, half_d)
        self.fc_silu = nn.Linear(half_d, half_d)

        bits = []
        for ch in owner_sign_text.encode("utf-8"):
            for i in range(7, -1, -1):
                bits.append(1.0 if ((ch >> i) & 1) == 1 else -1.0)
        while len(bits) < d_model:
            bits.append(1.0)
        target_signs = torch.tensor(bits[:d_model], dtype=torch.float32)
        self.register_buffer("deepipr_target_signs", target_signs)
        self.deepipr_passport_scale = nn.Parameter(target_signs * 1.0)

        with torch.no_grad():
            torch.manual_seed(2701)
            nn.init.orthogonal_(self.fc_gelu.weight, gain=2.2)
            nn.init.constant_(self.fc_gelu.bias, 0.35)
            nn.init.orthogonal_(self.fc_silu.weight, gain=2.2)
            nn.init.constant_(self.fc_silu.bias, -0.25)

        self.register_buffer("ethical_anchor_gelu", self.fc_gelu.weight.detach().clone())
        self.register_buffer("ethical_anchor_silu", self.fc_silu.weight.detach().clone())
        self.lock_ethical_weight_anchors()

    def lock_ethical_weight_anchors(self):
        with torch.no_grad():
            self.ethical_anchor_gelu.copy_(self.fc_gelu.weight.detach())
            self.ethical_anchor_silu.copy_(self.fc_silu.weight.detach())
        for p in self.parameters():
            p.requires_grad = False

    def verify_disk_license(self, root_dir: Path) -> str:
        """
        Verifica che LICENSE.md nella cartella del modello non sia stato rimosso o alterato
        per cancellare l'attribuzione a Thomas Seri o le clausole etiche.
        """
        lic_file = Path(root_dir) / "LICENSE.md"
        if not lic_file.exists():
            return "TAMPERED_MISSING_LICENSE_FILE"
        try:
            content = lic_file.read_text(encoding="utf-8", errors="ignore")
            for anchor in self.REQUIRED_DISK_LICENSE_ANCHORS:
                if anchor.lower() not in content.lower():
                    return f"TAMPERED_LICENSE_MISSING_{anchor}"
            return OFFICIAL_SENTINEL_LICENSE_SUMMARY
        except Exception:
            return "TAMPERED_UNREADABLE_LICENSE"

    def detect_malicious_finetuning_drift(self, max_allowed_drift: float = 0.05) -> Tuple[bool, float]:
        with torch.no_grad():
            drift_g = torch.norm(self.fc_gelu.weight - self.ethical_anchor_gelu).item()
            drift_s = torch.norm(self.fc_silu.weight - self.ethical_anchor_silu).item()
            total_drift = drift_g + drift_s
            signs = torch.sign(self.deepipr_passport_scale.detach())
            sign_acc = (signs == self.deepipr_target_signs).float().mean().item() * 100.0
            is_tampered = (total_drift > max_allowed_drift) or (sign_acc < 100.0)
            return is_tampered, total_drift

    @classmethod
    def _try_decode_hidden_payloads(cls, text: str) -> str:
        decoded_parts = [text]
        single_quoted = re.findall(r"(?:^|[\s(])'(.{1,45}?)'(?=$|[\s.,;!?)\]])", text)
        if len(single_quoted) >= 2:
            decoded_parts.append("".join(single_quoted))
        double_quoted = re.findall(r'"([^"]{1,45})"', text)
        if len(double_quoted) >= 2:
            decoded_parts.append("".join(double_quoted))
        for cand in re.findall(r"(?:^|[\s:original])([A-Za-z0-9+/]{12,}={0,2})(?=$|[\s.,;!?])", text):
            try:
                raw = base64.b64decode(cand, validate=False).decode("utf-8", errors="ignore")
                if len(raw) >= 4:
                    decoded_parts.append(raw)
            except Exception:
                pass
        for cand in re.findall(r"\b(?:[0-9a-fA-F]{2}){5,}\b", text):
            try:
                raw = binascii.unhexlify(cand).decode("utf-8", errors="ignore")
                if len(raw) >= 4:
                    decoded_parts.append(raw)
            except Exception:
                pass
        return " ".join(decoded_parts)

    @classmethod
    def _normalize_adversarial_text(cls, text: str) -> Tuple[str, str]:
        expanded = cls._try_decode_hidden_payloads(text)
        nfkd = unicodedata.normalize("NFKD", expanded)
        ascii_clean = "".join(c for c in nfkd if not unicodedata.combining(c))
        lowered = ascii_clean.lower().translate(cls.LEET_MAP)
        compact = re.sub(r"[^a-z0-9]", "", lowered)
        dedup = re.sub(r"(.)\1{2,}", r"\1", compact)
        return compact + " " + dedup, compact[::-1]

    def scan_security_and_ethics(self, prompt_text: str) -> Tuple[bool, str]:
        if not prompt_text:
            return True, "OK"

        # Decodifica prima eventuali payload Base64, Hex o Token-Split anche per il controllo GDPR
        expanded_text = self._try_decode_hidden_payloads(prompt_text)
        digits_only = re.sub(r"\D", "", expanded_text)
        digits_rev = digits_only[::-1]
        alnum_upper = re.sub(r"[^A-Z0-9]", "", expanded_text.upper())
        alnum_rev = alnum_upper[::-1]

        if re.search(r"IT\d{2}[A-Z]\d{22}", alnum_upper) or re.search(r"IT\d{2}[A-Z]\d{22}", alnum_rev):
            return False, "Violazione Licenza R08 Sez. 3.1 (Privacy GDPR): rilevato IBAN in chiaro/offuscato"
        if len(digits_only) >= 16 and (
            re.search(r"(?:4\d{15}|5[1-5]\d{14}|3[47]\d{13})", digits_only)
            or re.search(r"(?:4\d{15}|5[1-5]\d{14}|3[47]\d{13})", digits_rev)
        ):
            return False, "Violazione Licenza R08 Sez. 3.1 (Privacy GDPR): rilevata Carta di Credito"

        norm_fwd, norm_rev = self._normalize_adversarial_text(prompt_text)
        for benign in self.BENIGN_ALLOW_PHRASES:
            b_clean = re.sub(r"[^a-z0-9]", "", benign.lower().translate(self.LEET_MAP))
            norm_fwd = norm_fwd.replace(b_clean, "")

        for jb in self.JAILBREAK_ROOTS:
            jb_clean = re.sub(r"[^a-z0-9]", "", jb.lower().translate(self.LEET_MAP))
            if jb_clean in norm_fwd or jb_clean in norm_rev:
                return False, f"Violazione Licenza R08 (Jailbreak / Rimozione Autore Thomas Seri: '{jb}')"

        for root in self.FORBIDDEN_ROOTS:
            r_clean = re.sub(r"[^a-z0-9]", "", root.lower().translate(self.LEET_MAP))
            if r_clean in norm_fwd or r_clean in norm_rev:
                return False, f"Violazione Licenza R08 Sez. 3.1 (Uso Vietato: '{root}')"

        return True, "OK"

    def _get_live_scanner_bytecode_seal(self) -> bytes:
        try:
            fn_code = self.scan_security_and_ethics.__code__
            raw = (
                fn_code.co_code
                + repr(fn_code.co_consts).encode("utf-8")
                + repr(self.FORBIDDEN_ROOTS).encode("utf-8")
                + repr(self.JAILBREAK_ROOTS).encode("utf-8")
            )
            return raw
        except Exception:
            return b"TAMPERED_SCANNER"

    @classmethod
    def get_canonical_scanner_bytecode_seal(cls) -> bytes:
        fn_code = cls.scan_security_and_ethics.__code__
        return (
            fn_code.co_code
            + repr(fn_code.co_consts).encode("utf-8")
            + repr(cls.FORBIDDEN_ROOTS).encode("utf-8")
            + repr(cls.JAILBREAK_ROOTS).encode("utf-8")
        )

    def compute_nonlinear_wave(
        self,
        h_even: torch.Tensor,
        gate_lic: torch.Tensor,
        bias_lic: torch.Tensor
    ) -> torch.Tensor:
        u1 = F.gelu(self.fc_gelu(h_even) + bias_lic)
        u2 = F.silu(self.fc_silu(u1)) * gate_lic
        return 3.85 * u2 + 2.40 * torch.tanh(u1 * u2)

    def inject_geodesic_nonlinear_entanglement(
        self,
        h_clean: torch.Tensor,
        owner_passphrase: str = ""
    ) -> torch.Tensor:
        _, gate_0, bias_0, _ = derive_license_nonlinear_constants(
            license_text=OFFICIAL_SENTINEL_LICENSE_SUMMARY,
            d_model=self.d_model,
            scanner_bytecode=self.get_canonical_scanner_bytecode_seal(),
            owner_passphrase=owner_passphrase
        )
        gate_0 = gate_0.to(h_clean.device, dtype=h_clean.dtype)
        bias_0 = bias_0.to(h_clean.device, dtype=h_clean.dtype)

        h_even = h_clean[..., 0::2]
        h_odd = h_clean[..., 1::2]
        u1 = F.gelu(F.linear(h_even, self.ethical_anchor_gelu, self.fc_gelu.bias) + bias_0)
        u2 = F.silu(F.linear(u1, self.ethical_anchor_silu, self.fc_silu.bias)) * gate_0
        wave = 3.85 * u2 + 2.40 * torch.tanh(u1 * u2)

        entangled = torch.empty_like(h_clean)
        entangled[..., 0::2] = h_even
        entangled[..., 1::2] = h_odd + wave
        return entangled

    def forward(
        self,
        h_entangled: torch.Tensor,
        is_ethical: bool = True,
        license_text: str = OFFICIAL_SENTINEL_LICENSE_SUMMARY,
        owner_passphrase: str = ""
    ) -> torch.Tensor:
        is_tampered, _ = self.detect_malicious_finetuning_drift()
        if is_tampered:
            return -torch.sin(h_entangled * 17.3) * 4.0

        if not is_ethical:
            return -torch.tanh(h_entangled * 5.0)

        q_lic, gate_lic, bias_lic, _ = derive_license_nonlinear_constants(
            license_text=license_text,
            d_model=self.d_model,
            scanner_bytecode=self._get_live_scanner_bytecode_seal(),
            owner_passphrase=owner_passphrase
        )
        q_lic = q_lic.to(h_entangled.device, dtype=h_entangled.dtype)
        gate_lic = gate_lic.to(h_entangled.device, dtype=h_entangled.dtype)
        bias_lic = bias_lic.to(h_entangled.device, dtype=h_entangled.dtype)

        h_even = h_entangled[..., 0::2]
        h_odd_curved = h_entangled[..., 1::2]
        wave_live = self.compute_nonlinear_wave(h_even, gate_lic, bias_lic)

        restored = torch.empty_like(h_entangled)
        restored[..., 0::2] = h_even
        restored[..., 1::2] = h_odd_curved - wave_live

        sign_factor = torch.sign(self.deepipr_passport_scale) * self.deepipr_target_signs
        restored = restored * sign_factor
        return torch.matmul(restored, q_lic)
