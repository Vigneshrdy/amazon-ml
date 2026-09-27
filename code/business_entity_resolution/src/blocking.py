"""Candidate generation (blocking): per-country TF-IDF char-ngram top-K.

Country is a hard partition (we never compare across countries). Within a
country we vectorize a combined "name + address" field with char n-grams so a
match survives typos AND is found via a shared address even when the name is
mangled or in another script. sparse_dot_topn keeps only the top-K neighbours
per S1 record, which is what bounds the candidate set.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from common import norm_name, norm_addr


def combined_text(names, addrs):
    return [f"{n} {a}".strip() for n, a in zip(names, addrs)]


def block_country(s1_df, tgt_df, top_k=25, min_sim=0.30):
    """Return list of (s1_idx, tgt_idx, sim) for one country / one target source.

    Indices are positional into the passed frames.
    """
    if len(s1_df) == 0 or len(tgt_df) == 0:
        return []
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2)
    corpus = combined_text(tgt_df["_nm"].tolist(), tgt_df["_ad"].tolist())
    Xt = vec.fit_transform(corpus)
    Xs = vec.transform(combined_text(s1_df["_nm"].tolist(), s1_df["_ad"].tolist()))
    # top_k neighbours of each S1 row among target rows, cosine (tfidf is L2-normed)
    C = sp_matmul_topn(Xs, Xt.T, top_n=top_k, threshold=min_sim, sort=True,
                       n_threads=-1)
    C = C.tocoo()
    return list(zip(C.row.tolist(), C.col.tolist(), C.data.tolist()))
