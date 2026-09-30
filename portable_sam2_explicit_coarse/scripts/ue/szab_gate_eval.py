"""MHBEV-FUS-20260929-SZAB gate evaluation (Appendix V.4, mechanical).

Reads both arms' artifacts, evaluates SZAB_INJECT_PASS:
 (i)   AP_s(B, last) > AP_s(A, last) and small-mask-F1 B-A paired bootstrap
       (10k, seed 20260936) CI lower > 0 at the last epoch;
 (ii)  AP(B)-AP(A) >= -0.003 and BF1(B)-BF1(A) >= -0.003 at the last epoch;
 (iii) B arm injection active: gamma(last) deviates from init 0.25 by >= 20% and
       engagement (epoch mean-square |delta|^2/|P2|^2 ratio) > 0 at every epoch;
 (iv)  manifests share git_commit, ckpt SHA256s, seed triple; init_state_hash_shared
       identical; and neither manifest used smoke limits.

Writes summary.json + prints the verdict line. Read-only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

BOOT_SEED = 20260936
DRAWS = 10000
GAMMA_INIT = 0.25


def paired_ci(a_rows, b_rows, key):
    common = sorted({r["scene"] for r in a_rows} & {r["scene"] for r in b_rows})
    amap = {r["scene"]: r[key] for r in a_rows}
    bmap = {r["scene"]: r[key] for r in b_rows}
    d = np.array([bmap[s] - amap[s] for s in common], dtype=np.float64)
    idx = np.random.RandomState(BOOT_SEED).randint(0, len(d), size=(DRAWS, len(d)))
    means = d[idx].mean(1)
    return dict(mean=float(d.mean()),
                ci=[float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                improved=int((d > 0).sum()), n=len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a", required=True)
    ap.add_argument("--arm-b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    A, B = Path(args.arm_a), Path(args.arm_b)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    hA = json.load(open(A / "history.json"))
    hB = json.load(open(B / "history.json"))
    mA = json.load(open(A / "manifest.json"))
    mB = json.load(open(B / "manifest.json"))
    lastA, lastB = hA[-1], hB[-1]
    ep = lastA["epoch"]
    assert lastB["epoch"] == ep, f"epoch mismatch A={lastA['epoch']} B={lastB['epoch']}"

    sfA = json.load(open(A / f"per_scene_smallf1_epoch{ep}.json"))
    sfB = json.load(open(B / f"per_scene_smallf1_epoch{ep}.json"))
    ci_f1 = paired_ci(sfA, sfB, "f1")

    d_ap_s = lastB["AP_s"] - lastA["AP_s"]
    d_ap = lastB["AP"] - lastA["AP"]
    d_bf1 = lastB["boundary_f1"] - lastA["boundary_f1"]
    gB = lastB.get("inject_gamma")
    engB = [h.get("inject_engagement_ms_ratio", 0.0) for h in hB]

    checks = dict(
        gate_i_ap_s_positive=bool(d_ap_s > 0),
        gate_i_smallf1_ci_lower_positive=bool(ci_f1["ci"][0] > 0),
        gate_ii_ap_nondegradation=bool(d_ap >= -0.003),
        gate_ii_bf1_nondegradation=bool(d_bf1 >= -0.003),
        gate_iii_gamma_deviation_20pct=bool(gB is not None and abs(gB - GAMMA_INIT) >= 0.2 * GAMMA_INIT),
        gate_iii_engagement_positive=bool(len(engB) > 0 and all(e > 0 for e in engB)),
        gate_iv_manifest_invariants=bool(
            mA["git_commit"] == mB["git_commit"]
            and mA["single_ckpt_sha256"] == mB["single_ckpt_sha256"]
            and mA["dual_ckpt_sha256"] == mB["dual_ckpt_sha256"]
            and mA["seeds"] == mB["seeds"]
            and mA["init_state_hash_shared"] == mB["init_state_hash_shared"]
            and not mA["limit_train_batches"] and not mA["limit_val_scenes"]
            and not mB["limit_train_batches"] and not mB["limit_val_scenes"]),
    )
    verdict = "SZAB_INJECT_PASS" if all(checks.values()) else "SZAB_INJECT_NO_SMALL_GAIN"

    summary = dict(
        verdict=verdict, checks=checks, epoch=ep,
        arm_a=dict(AP_s=lastA["AP_s"], AP=lastA["AP"], BF1=lastA["boundary_f1"],
                   small_f1=lastA["small_f1"], history=hA),
        arm_b=dict(AP_s=lastB["AP_s"], AP=lastB["AP"], BF1=lastB["boundary_f1"],
                   small_f1=lastB["small_f1"], gamma=gB,
                   engagement_ms_ratio_per_epoch=engB, history=hB),
        paired=dict(
            small_f1_B_minus_A=ci_f1,
            AP_s_delta=float(d_ap_s), AP_delta=float(d_ap), BF1_delta=float(d_bf1)),
        manifest_a=mA, manifest_b=mB)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"epoch {ep}: AP_s A={lastA['AP_s']:.4f} B={lastB['AP_s']:.4f} d={d_ap_s:+.4f}")
    print(f"small-F1 B-A: {ci_f1['mean']:+.4f} CI[{ci_f1['ci'][0]:+.4f},{ci_f1['ci'][1]:+.4f}] "
          f"improved {ci_f1['improved']}/{ci_f1['n']}")
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
