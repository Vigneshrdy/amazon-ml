# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We resolve entities with a classic, precision-first ER pipeline: country-partitioned
character-n-gram TF-IDF blocking over a combined *name + address* field, a LightGBM
classifier on cheap string/address similarity features, and a decision threshold
tuned directly for macro F0.5. Key innovation: transliterating every script to Latin
before blocking so Devanagari/Tamil business names match their English counterparts,
and blocking on name **and** address jointly so a record is still recovered when its
name is mangled but its address survives.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the training data surfaced the noise the pipeline must survive:
- **Cross-script names.** Source-2/3 India records frequently carry the name in
  Devanagari or Tamil (`ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்` ↔ `Raj Investments`) while
  addresses stay largely Latin. Pure name string-similarity fails here.
- **Severe name typos / substitutions.** A true match sometimes shares almost no
  name tokens (`Maure Williams Colombier` ↔ `Dréxkor`, `maurewilliamscolombier.com`)
  and is only recoverable via the address.
- **Address variation.** Abbreviations (`Rd`/`Road`, `St`/`Street`), reordered
  components, `null` placeholders, missing PIN/state, transliterated state names
  (`Tamil Nadu`/`தமிழ்நாடு`/`TN`).
- **Match cardinality.** Most Source-1 entities have 2–4 matches; ~5.6% are
  singletons (must be predicted as empty for full credit).
- **Open country set.** Test adds **France** (~24% of test S1), absent from train.
  The pipeline never branches on the country value — it is used only as a hard
  blocking partition — so an unseen label costs nothing.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (supervised pairwise matching)
**Core Innovation:** Script-transliterated, name+address-combined char-n-gram
blocking (robust to script and typos), with an F0.5-tuned threshold.

Records are normalized once; candidates are generated per country; a gradient-boosted
tree scores each candidate; a single threshold chosen for macro F0.5 turns scores
into the final match lists.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Latin-transliterated (Unidecode), suffix/abbreviation-folded
  *name + address* text, vectorized as TF-IDF char word-boundary 3–4-grams; top-25
  cosine neighbours per Source-1 record (threshold 0.30) via `sparse_dot_topn`,
  computed **within each country** and separately against Source-2 and Source-3.
- **Candidate pairs generated:** reported by `run.py` (`candidates …`) at run time;
  `candidate_pairs.tsv` is exactly the set fed to the classifier.
- **How true matches were preserved:** combining name and address in one field
  recovers matches that either signal alone would miss (script-swapped names via
  address tokens, address-less records via name). Measured blocking recall ceiling
  ≈ **0.97** on held-out training data.

---

## 4. Matching Model

**Features used:**
- **Name:** TF-IDF cosine (from blocking), rapidfuzz token_sort / token_set /
  partial ratios, token Jaccard.
- **Address:** token_set ratio, token Jaccard, both-address-present flag.
- **Other:** numeric-token (house number / PIN) Jaccard, name length ratio,
  Source-2-vs-3 indicator.

**Model type:** LightGBM (gradient-boosted decision trees), 300 trees,
`learning_rate=0.05`, `num_leaves=63`. Positives are ground-truth matches among the
blocked candidates; negatives are the remaining candidates (natural hard negatives).
**Threshold selection:** grid search over 0.30–0.95 maximising **macro F0.5** on a
20% Source-1 holdout — F0.5's precision weighting pushes the threshold high (~0.72
in development), suppressing false merges.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** ~0.96 on the development holdout (measured against
  ground-truth targets present in the loaded data; the full-data private-set figure
  will be lower but is bounded above by the ~0.97 blocking recall).
- **Common false positives:** distinct businesses sharing an address (co-located
  tenants, chains) or near-identical common names — mitigated by the high threshold.
- **Common false negatives:** matches with both a rewritten name and a heavily
  abbreviated/landmark-only address, where char-ngram overlap drops below the
  blocking cutoff.

---

## 6. Conclusion

A transliteration-first, name+address char-ngram blocker plus an F0.5-tuned LightGBM
matcher gives strong precision-weighted resolution with no external data and modest
compute. The largest remaining lever is blocking recall (a phonetic or PIN second
key) and per-country thresholds.

---

## Appendix

### A. Code Artefacts

Runnable pipeline under `code/business_entity_resolution/`:
- `src/common.py` — normalization, transliteration, pairwise features (+ self-check).
- `src/blocking.py` — per-country TF-IDF char-ngram top-K candidate generation.
- `src/run.py` — `train` (fit + F0.5 threshold) and `predict` (write both TSVs).
- `README.md` — exact reproduce commands; `requirements.txt` — pinned deps.

Entry points regenerate `output/matching_results.tsv` and `output/candidate_pairs.tsv`
from the provided data (see README).

### B. Additional Results

Blocking recall ceiling and candidate counts are printed by `run.py train`.
