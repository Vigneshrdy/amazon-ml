"""Shared normalization + pairwise similarity features for the ER pipeline.

Country-agnostic on purpose (test set adds France, unseen in train): we never
branch on the country label, only use it as a hard blocking key.
"""
import re
import numpy as np
from unidecode import unidecode
from rapidfuzz import fuzz

# Legal-suffix / abbreviation folding. Applied as whole-word replacements after
# transliteration + lowercasing so "Pvt Ltd" == "private limited", "Corp" ==
# "corporation", "&" == "and", etc.
_NAME_MAP = {
    "corp": "corporation", "co": "company", "inc": "incorporated",
    "ltd": "limited", "pvt": "private", "llp": "llp", "llc": "llc",
    "intl": "international", "&": "and", "mfg": "manufacturing",
    "svc": "service", "svcs": "service", "assoc": "associates",
}
_ADDR_MAP = {
    "rd": "road", "st": "street", "ave": "avenue", "ave.": "avenue",
    "blvd": "boulevard", "ln": "lane", "dr": "drive", "hwy": "highway",
    "sq": "square", "apt": "apartment", "fl": "floor", "bldg": "building",
    "opp": "opposite", "nr": "near", "no": "number", "sec": "sector",
    "n": "north", "s": "south", "e": "east", "w": "west",
}
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def _fold(text, mapping):
    # transliterate (Devanagari/Tamil/accents -> latin) then fold tokens
    text = unidecode(text or "").lower().replace("&", " and ")
    text = _PUNCT.sub(" ", text)
    toks = [mapping.get(t, t) for t in text.split()]
    return _WS.sub(" ", " ".join(toks)).strip()


def norm_name(s):
    return _fold(s, _NAME_MAP)


def norm_addr(s):
    # drop the literal "null" tokens the sources use for missing components
    s = re.sub(r"\bnull\b", " ", (s or ""), flags=re.I)
    return _fold(s, _ADDR_MAP)


def digits(s):
    """Set of numeric tokens (house numbers, PIN codes) — strong ER signal."""
    return set(re.findall(r"\d+", s or ""))


def _jacc(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def pair_features(nm1, ad1, dg1, nm2, ad2, dg2, block_sim, is_s3):
    """Cheap similarity features for one (S1, candidate) pair.

    Inputs are pre-normalized strings + precomputed digit sets so the caller
    normalizes each record once, not once per pair.
    """
    return [
        block_sim,                                   # tfidf cosine from blocking
        fuzz.token_sort_ratio(nm1, nm2) / 100.0,
        fuzz.token_set_ratio(nm1, nm2) / 100.0,
        fuzz.partial_ratio(nm1, nm2) / 100.0,
        _jacc(nm1.split(), nm2.split()),
        fuzz.token_set_ratio(ad1, ad2) / 100.0,
        _jacc(ad1.split(), ad2.split()),
        _jacc(dg1, dg2),                             # numeric-token overlap
        1.0 if (ad1 and ad2) else 0.0,               # both addresses present
        abs(len(nm1) - len(nm2)) / (max(len(nm1), len(nm2)) + 1),
        1.0 if is_s3 else 0.0,                       # source indicator S2/S3
    ]


FEATURE_NAMES = [
    "block_sim", "name_token_sort", "name_token_set", "name_partial",
    "name_jaccard", "addr_token_set", "addr_jaccard", "digit_jaccard",
    "both_addr", "name_len_ratio", "is_s3",
]


def demo():
    a = norm_name("Pvt Ltd Raj & Co.")
    assert "private" in a and "and" in a and "company" in a, a
    assert norm_name("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்"), "transliteration empty"
    assert _jacc({"1", "2"}, {"2", "3"}) == 1 / 3
    f = pair_features("raj", "1 main rd", {"1"}, "raj", "1 main road", {"1"},
                      0.9, False)
    assert len(f) == len(FEATURE_NAMES)
    print("common.py self-check OK:", a)


if __name__ == "__main__":
    demo()
