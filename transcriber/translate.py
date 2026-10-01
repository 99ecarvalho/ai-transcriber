# Copyright (c) 2026 Eduardo Correia <ecorreia@apliant.com.br>
#
# This file is part of ai-transcriber. It is free software, licensed under the
# GNU Lesser General Public License v3.0 or later. See COPYING.LESSER and
# COPYING for details.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Text translation between languages, run on CTranslate2 (the engine faster-whisper uses).

Whisper itself only translates *into English*. This adds any-to-any translation
of the transcribed text. Two model families are supported, detected from the
model's vocabulary:

- M2M100 (MIT license), the default.
- NLLB-200 (CC-BY-NC-4.0: non-commercial use only), opt-in via TRANSLATION_MODEL.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

# Whisper language code -> FLORES-200 code used by NLLB. Entries the loaded
# model doesn't know are ignored (checked against its vocabulary).
NLLB_CODES = {
    "af": "afr_Latn", "am": "amh_Ethi", "ar": "arb_Arab", "as": "asm_Beng", "az": "azj_Latn",
    "ba": "bak_Cyrl", "be": "bel_Cyrl", "bg": "bul_Cyrl", "bn": "ben_Beng", "bo": "bod_Tibt",
    "bs": "bos_Latn", "ca": "cat_Latn", "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn",
    "de": "deu_Latn", "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn", "et": "est_Latn",
    "eu": "eus_Latn", "fa": "pes_Arab", "fi": "fin_Latn", "fo": "fao_Latn", "fr": "fra_Latn",
    "gl": "glg_Latn", "gu": "guj_Gujr", "ha": "hau_Latn", "he": "heb_Hebr", "hi": "hin_Deva",
    "hr": "hrv_Latn", "ht": "hat_Latn", "hu": "hun_Latn", "hy": "hye_Armn", "id": "ind_Latn",
    "is": "isl_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "jw": "jav_Latn", "ka": "kat_Geor",
    "kk": "kaz_Cyrl", "km": "khm_Khmr", "kn": "kan_Knda", "ko": "kor_Hang", "lb": "ltz_Latn",
    "ln": "lin_Latn", "lo": "lao_Laoo", "lt": "lit_Latn", "lv": "lvs_Latn", "mg": "plt_Latn",
    "mi": "mri_Latn", "mk": "mkd_Cyrl", "ml": "mal_Mlym", "mn": "khk_Cyrl", "mr": "mar_Deva",
    "ms": "zsm_Latn", "mt": "mlt_Latn", "my": "mya_Mymr", "ne": "npi_Deva", "nl": "nld_Latn",
    "nn": "nno_Latn", "no": "nob_Latn", "oc": "oci_Latn", "pa": "pan_Guru", "pl": "pol_Latn",
    "ps": "pbt_Arab", "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "sa": "san_Deva",
    "sd": "snd_Arab", "si": "sin_Sinh", "sk": "slk_Latn", "sl": "slv_Latn", "sn": "sna_Latn",
    "so": "som_Latn", "sq": "als_Latn", "sr": "srp_Cyrl", "su": "sun_Latn", "sv": "swe_Latn",
    "sw": "swh_Latn", "ta": "tam_Taml", "te": "tel_Telu", "tg": "tgk_Cyrl", "th": "tha_Thai",
    "tk": "tuk_Latn", "tl": "tgl_Latn", "tr": "tur_Latn", "tt": "tat_Cyrl", "uk": "ukr_Cyrl",
    "ur": "urd_Arab", "uz": "uzn_Latn", "vi": "vie_Latn", "yi": "ydd_Hebr", "yo": "yor_Latn",
    "zh": "zho_Hans", "yue": "yue_Hant",
}
# Whisper codes that M2M100 spells differently.
M2M100_ALIASES = {"jw": "jv"}


def _read_vocabulary(model_dir: Path) -> set[str]:
    json_path = model_dir / "shared_vocabulary.json"
    if json_path.exists():
        return set(json.loads(json_path.read_text(encoding="utf-8")))
    txt_path = model_dir / "shared_vocabulary.txt"
    if txt_path.exists():
        return {line.rstrip("\n") for line in txt_path.read_text(encoding="utf-8").splitlines()}
    raise ValueError(f"{model_dir} has no shared_vocabulary.json/.txt — is it a CTranslate2 model?")


def resolve_model_dir(model: str) -> Path:
    """A local directory, or a Hugging Face repo id downloaded into the HF cache."""
    if os.path.isdir(model):
        return Path(model)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model))


class TextTranslator:
    def __init__(self, model_dir: Path, device: str, compute_type: str):
        import ctranslate2
        import sentencepiece

        self.vocabulary = _read_vocabulary(model_dir)
        if "eng_Latn" in self.vocabulary:
            self.family = "nllb"
        elif "__en__" in self.vocabulary:
            self.family = "m2m100"
        else:
            raise ValueError(f"{model_dir} is neither an M2M100 nor an NLLB model")
        self.sp = sentencepiece.SentencePieceProcessor(model_file=str(model_dir / "sentencepiece.bpe.model"))
        self.model = ctranslate2.Translator(str(model_dir), device=device, compute_type=compute_type)

    def lang_token(self, code: str) -> str | None:
        code = code.lower()
        if self.family == "nllb":
            token = NLLB_CODES.get(code)
        else:
            token = f"__{M2M100_ALIASES.get(code, code)}__"
        return token if token in self.vocabulary else None

    def supports(self, code: str) -> bool:
        return self.lang_token(code) is not None

    def translate(self, texts: list[str], source: str, target: str) -> list[str]:
        src, tgt = self.lang_token(source), self.lang_token(target)
        if src is None or tgt is None:
            raise ValueError(f"translation {source} -> {target} is not supported by this model")
        if source == target:
            return list(texts)
        # Both model families are trained on single sentences and tend to drop
        # all but one sentence of a longer input, so translate sentence by sentence.
        pieces = [(i, sentence) for i, t in enumerate(texts) for sentence in split_sentences(t)]
        out: list[list[str]] = [[] for _ in texts]
        if not pieces:
            return ["" for _ in texts]
        batch = [[src, *self.sp.encode(sentence, out_type=str), "</s>"] for _, sentence in pieces]
        results = self.model.translate_batch(
            batch, target_prefix=[[tgt]] * len(batch), beam_size=4, max_decoding_length=512
        )
        for (i, _), r in zip(pieces, results, strict=True):
            out[i].append(self.sp.decode(r.hypotheses[0][1:]).strip())
        joiner = "" if target in NO_SPACE_LANGUAGES else " "
        return [joiner.join(parts) for parts in out]


# Scripts written without spaces between sentences.
NO_SPACE_LANGUAGES = {"zh", "ja", "yue", "th", "lo", "km", "my", "bo"}
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in (part.strip() for part in _SENTENCE_END.split(text.strip())) if s]
