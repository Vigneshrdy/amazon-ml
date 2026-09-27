# Business Entity Resolution — Amazon ML Challenge 2026

Given business records from **3 independent, noisy sources** with no shared
identifiers, decide which Source-2/Source-3 records refer to the same real-world
business as each Source-1 (reference) entity.

Full challenge spec: [`PROBLEM.md`](PROBLEM.md).

## Approach

Precision-first ER pipeline:

**Normalize → block → similarity features → LightGBM → F0.5-tuned threshold.**
No external data — only the provided files.

- **Transliterate any script to Latin** (Unidecode) so Devanagari/Tamil names
  match their English counterparts.
- **Block on `name + address` jointly**, per country, via char-n-gram TF-IDF —
  a record is still recovered when its name is mangled but its address survives.
- **Country is an open set** (test adds France, unseen in train): used only as a
  hard blocking partition, never branched on.

Solution details: [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md).

## Layout

```
dataset/          train/ + test/ TSVs (gitignored — 2.5 GB)
code/business_entity_resolution/
  src/            common.py, blocking.py, run.py
  artifacts/      trained model.txt + threshold.txt (gitignored)
  requirements.txt
output/           generated submission TSVs (gitignored)
utils/            validate_submission.py
PROBLEM.md        challenge problem statement
Documentation_template.md   submission write-up
```

## Quickstart

```bash
pip install -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution/src

# train + validate → ../artifacts
python3 run.py train   --data ../../../dataset/train --out ../artifacts

# predict → student_resource/output
python3 run.py predict --data ../../../dataset/test  --artifacts ../artifacts \
    --out ../../../output
```

Validate before upload (from repo root):

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

> Files are **tab-separated** — always read with `sep="\t"`. The full test set
> (~1.7M S1, ~10M S2+S3) needs a large-RAM instance; use `--sample`/`--max-rows`
> for dev.
