# Business Entity Resolution — Amazon ML Challenge 2026

Blocking → similarity features → LightGBM → F0.5-tuned threshold. No external
data or lookups; only the provided training/test files are used.

## Pipeline

1. **Normalize** (`common.py`): transliterate any script to Latin (Unidecode,
   handles Devanagari/Tamil names), lowercase, fold legal suffixes and address
   abbreviations (`Pvt Ltd`→`private limited`, `Rd`→`road`, `&`→`and`), strip
   punctuation, drop `null` tokens.
2. **Blocking** (`blocking.py`): per **country** (hard partition — never compare
   across countries; country is an open set, France included), TF-IDF char
   `word-boundary` 3–4-grams over a combined `name + address` field, keep the
   top-25 neighbours per Source-1 record (cosine ≥ 0.30) via `sparse_dot_topn`.
   The combined field means a record is still found via a shared address when the
   name is mangled or in another script.
3. **Features** (`common.py:pair_features`): tfidf cosine, rapidfuzz
   token_sort/token_set/partial name ratios, name token Jaccard, address
   token_set + Jaccard, numeric-token (house/PIN) Jaccard, both-address flag,
   name length ratio, S2/S3 source flag.
4. **Matcher** (`run.py`): LightGBM binary classifier. Positives = ground-truth
   matches among candidates; negatives = the rest of the blocked candidates.
   The decision threshold is grid-searched to maximise **macro F0.5** on a 20%
   Source-1 holdout (precision-heavy, so it lands high — ~0.72 in dev).

Every Source-1 entity gets exactly one output row (empty when no candidate
clears the threshold), satisfying the singleton rule.

## Reproduce

```bash
pip install -r requirements.txt
cd src

# train + validate, saves model.txt + threshold.txt into ../artifacts
python3 run.py train   --data ../../../dataset/train --out ../artifacts

# generate submission files into student_resource/output
python3 run.py predict --data ../../../dataset/test  --artifacts ../artifacts \
    --out ../../../output
```

`--sample N` (Source-1 entities) and `--max-rows N` (rows per source file) cap
memory for dev on a small box; **omit both for the full run**. The full test set
(~1.7M S1, ~10M S2+S3) needs a large-RAM instance — the char-ngram TF-IDF matrix
per country is the memory driver.

Validate before uploading (from `student_resource/`):

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Self-checks

`python3 src/common.py` runs an assert-based check of normalization + features.

## Known ceilings

- `# ponytail:` blocking is single-key char-ngram; a phonetic/PIN second key
  would raise the ~0.97 recall ceiling further. Add if recall becomes binding.
- Threshold is global; per-country thresholds could squeeze more F0.5.
