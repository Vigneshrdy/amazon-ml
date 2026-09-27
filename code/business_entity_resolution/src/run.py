"""End-to-end ER pipeline: blocking -> features -> LightGBM -> F0.5 threshold.

Usage (from code/business_entity_resolution/src):

  # train + validate on a holdout of the training data, save model+threshold
  python3 run.py train --data ../../../dataset/train --out ../artifacts \
      [--sample 40000] [--max-rows 400000]

  # generate submission files for the test set using the saved model
  python3 run.py predict --data ../../../dataset/test \
      --artifacts ../artifacts --out ../../../output

--sample caps the number of Source-1 entities used (dev/validation on a box
with little RAM); --max-rows caps rows loaded per source file. Omit both on the
full AWS run.
"""
import argparse
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import norm_name, norm_addr, digits, pair_features, FEATURE_NAMES
from blocking import block_country

TOP_K = 25
MIN_SIM = 0.30


def load_source(path, max_rows=None):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                     nrows=max_rows)
    df["_nm"] = df["business_name"].map(norm_name)
    df["_ad"] = df["business_address"].map(norm_addr)
    df["_dg"] = (df["business_name"] + " " + df["business_address"]).map(digits)
    return df


def load_gt(path):
    gt = {}
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for sid, ids in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        gt[sid] = set(x for x in ids.split(",") if x)
    return gt


def gen_candidates(s1, s2, s3):
    """Yield (s1_id, tgt_id, feature_list, sim) over all country blocks.

    One pass per country per target source. Records are normalized once (on the
    frame); features are built here per candidate pair.
    """
    for src_df, is_s3 in ((s2, False), (s3, True)):
        for country in s1["country"].unique():
            a = s1[s1["country"] == country].reset_index(drop=True)
            b = src_df[src_df["country"] == country].reset_index(drop=True)
            pairs = block_country(a, b, TOP_K, MIN_SIM)
            if not pairs:
                continue
            an, aa, ad, aid = (a["_nm"].to_numpy(), a["_ad"].to_numpy(),
                               a["_dg"].to_numpy(), a["entity_id"].to_numpy())
            bn, ba, bd, bid = (b["_nm"].to_numpy(), b["_ad"].to_numpy(),
                               b["_dg"].to_numpy(), b["entity_id"].to_numpy())
            for i, j, sim in pairs:
                feats = pair_features(an[i], aa[i], ad[i], bn[j], ba[j], bd[j],
                                      sim, is_s3)
                yield aid[i], bid[j], feats, sim


def collect(s1, s2, s3):
    """Group candidates by s1_id -> (tgt_ids, feature_matrix)."""
    by = {}
    for sid in s1["entity_id"]:
        by[sid] = ([], [])
    for sid, tid, feats, _ in gen_candidates(s1, s2, s3):
        by[sid][0].append(tid)
        by[sid][1].append(feats)
    return by


def f05(pred, truth):
    pred, truth = set(pred), set(truth)
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(by, scores_by, gt, t):
    tot = 0.0
    for sid, (tids, _) in by.items():
        sc = scores_by[sid]
        pred = [tids[k] for k in range(len(tids)) if sc[k] >= t]
        tot += f05(pred, gt.get(sid, set()))
    return tot / len(by)


def cmd_train(args):
    os.makedirs(args.out, exist_ok=True)
    s1 = load_source(f"{args.data}/train_source1.tsv", args.max_rows)
    if args.sample:
        s1 = s1.sample(min(args.sample, len(s1)), random_state=0).reset_index(drop=True)
    s2 = load_source(f"{args.data}/train_source2.tsv", args.max_rows)
    s3 = load_source(f"{args.data}/train_source3.tsv", args.max_rows)
    gt = load_gt(f"{args.data}/train_ground_truth.tsv")

    # split S1 entities into train/val
    rng = np.random.default_rng(0)
    ids = s1["entity_id"].to_numpy()
    val_mask = rng.random(len(ids)) < 0.2
    val_ids = set(ids[val_mask])

    by = collect(s1, s2, s3)

    # Restrict GT to targets actually present in the loaded data so local
    # (--max-rows) validation measures the model, not truncated-away matches.
    # On the full run seen == all target ids, so this is a no-op.
    seen = set(s2["entity_id"]) | set(s3["entity_id"])
    gt_seen = {sid: {x for x in ids if x in seen} for sid, ids in gt.items()}

    tp = fn = 0
    for sid, (tids, _) in by.items():
        truth = gt_seen.get(sid, set())
        cand = set(tids)
        tp += len(truth & cand); fn += len(truth - cand)
    print(f"blocking recall ceiling: {tp/(tp+fn+1e-9):.4f}  "
          f"(candidates {sum(len(v[0]) for v in by.values())})")

    # training rows from TRAIN split only
    X, y = [], []
    for sid, (tids, feats) in by.items():
        if sid in val_ids:
            continue
        truth = gt_seen.get(sid, set())
        for tid, f in zip(tids, feats):
            X.append(f); y.append(1 if tid in truth else 0)
    X, y = np.array(X, dtype=np.float32), np.array(y)
    print(f"train pairs: {len(y)}  positives: {y.sum()}")
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               num_leaves=63, subsample=0.8,
                               colsample_bytree=0.8, n_jobs=-1)
    model.fit(X, y)

    # score val candidates, tune threshold for macro F0.5
    val_by = {sid: v for sid, v in by.items() if sid in val_ids}
    scores_by = {}
    for sid, (tids, feats) in val_by.items():
        scores_by[sid] = (model.predict_proba(np.array(feats, dtype=np.float32))[:, 1]
                          if feats else np.array([]))
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.30, 0.96, 0.02):
        f = macro_f05(val_by, scores_by, gt_seen, t)
        if f > best_f:
            best_f, best_t = f, float(t)
    print(f"best threshold {best_t:.2f}  val macro F0.5 = {best_f:.4f}")
    model.booster_.save_model(f"{args.out}/model.txt")
    with open(f"{args.out}/threshold.txt", "w") as fh:
        fh.write(str(best_t))
    print("saved model + threshold to", args.out)


def _write_tsv(path, header2, rows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"source1_entity_id\t{header2}\n")
        for sid, ids in rows:
            fh.write(f"{sid}\t{','.join(ids)}\n")


def cmd_predict(args):
    os.makedirs(args.out, exist_ok=True)
    s1 = load_source(f"{args.data}/test_source1.tsv", args.max_rows)
    s2 = load_source(f"{args.data}/test_source2.tsv", args.max_rows)
    s3 = load_source(f"{args.data}/test_source3.tsv", args.max_rows)
    model = lgb.Booster(model_file=f"{args.artifacts}/model.txt")
    t = float(open(f"{args.artifacts}/threshold.txt").read().strip())

    by = collect(s1, s2, s3)
    cand_rows, match_rows = [], []
    for sid in s1["entity_id"]:            # every S1 entity gets exactly one row
        tids, feats = by[sid]
        cand_rows.append((sid, tids))
        if feats:
            sc = model.predict(np.array(feats, dtype=np.float32))
            matched = [tids[k] for k in range(len(tids)) if sc[k] >= t]
        else:
            matched = []
        match_rows.append((sid, matched))
    _write_tsv(f"{args.out}/candidate_pairs.tsv", "candidate_entity_ids", cand_rows)
    _write_tsv(f"{args.out}/matching_results.tsv", "matched_entity_ids", match_rows)
    print(f"wrote {len(match_rows)} rows to {args.out} (threshold {t})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train"); tr.set_defaults(fn=cmd_train)
    tr.add_argument("--data", required=True); tr.add_argument("--out", required=True)
    tr.add_argument("--sample", type=int, default=0)
    tr.add_argument("--max-rows", type=int, default=None)
    pr = sub.add_parser("predict"); pr.set_defaults(fn=cmd_predict)
    pr.add_argument("--data", required=True); pr.add_argument("--artifacts", required=True)
    pr.add_argument("--out", required=True); pr.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
