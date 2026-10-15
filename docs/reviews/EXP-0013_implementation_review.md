# EXP-0013 implementation review (independent audit, pre-launch)

Audit target: HEAD = `9ab3798` "feat(exp): EXP-0013 L0-only fusion arm completes the {P2}x{P3} factorial"
(branch `codex/dual-stream-dev`, worktree `/home/wangcheng/project/new/.worktrees/dual-stream-dev`,
audit date 2026-10-15). All claims below were recomputed independently; no tracked file was
modified except this report.

## 1. Single-variable property — OK

`diff configs/ue_whu_a3sgm_512_dualv2.py configs/ue_whu_a3sgm_512_dualv2_l0.py` (cwd
`portable_sam2_explicit_coarse/`) yields exactly two hunks:

- lines 1–3: module docstring (dualv2 → dualv2-L0 description);
- line 22: `'target_levels': [0, 1]` → `'target_levels': [0]`.

Nothing else differs. `git show 9ab3798 --name-status --format=` lists exactly the five
expected files and nothing else:

```
M  docs/EXP-0008_preregistration.md
M  docs/EXP_REGISTRY.md
A  portable_sam2_explicit_coarse/configs/ue_whu_a3sgm_512_dualv2_l0.py
A  portable_sam2_explicit_coarse/scripts/ue/run_exp0013_ddp4.sh
M  portable_sam2_explicit_coarse/ue/contract.py
```

The `ue/contract.py` change is a single line: the new registry entry
`"ue-whu-a3sgm-512-dualv2-l0-v1": "configs/ue_whu_a3sgm_512_dualv2_l0.py"` in
`BASELINE_CONFIGS`.

## 2. Fingerprint + registry — OK

With cwd `portable_sam2_explicit_coarse/`,
`SAM2_CKPT=/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt`, interpreter
`/data2/wangcheng/envs/cvt2/bin/python` (same computation as the wrapper gate):

```
fingerprint: 277a95dd4d005676bce2d0da423e1d0d84e48abed03f254c5f85b4a223635834
pinned:      277a95dd4d005676bce2d0da423e1d0d84e48abed03f254c5f85b4a223635834  -> match
arch_id: r1_c4_rd_pafpn_coarse_points_box_raw_detach_emb64_udprk64
```

`ue.contract.baseline_id_for` over all five configs:

```
configs/ue_whu_a3sgm_512.py            -> ue-whu-a3sgm-512-v1         OK
configs/ue_whu_a3sgm_512_dual.py       -> ue-whu-a3sgm-512-dual-v1    OK
configs/ue_whu_a3sgm_512_dualv2.py     -> ue-whu-a3sgm-512-dualv2-v1  OK
configs/ue_whu_a3sgm_512_dualv2_lora.py-> ue-whu-a3sgm-512-duallora-v1 OK
configs/ue_whu_a3sgm_512_dualv2_l0.py  -> ue-whu-a3sgm-512-dualv2-l0-v1 OK
all fingerprints distinct: True
  5305b315…/4ccb42b8…/d26f1f97…/50f6b097…/277a95dd…
```

Each config resolves only to its own identity; no cross-match (the comparison is exact
contract equality, and all five fingerprints are distinct).

## 3. Wrapper audit — OK

`diff scripts/ue/run_exp0011_ddp4.sh scripts/ue/run_exp0013_ddp4.sh` changes only: the
header comment (0013's is the corrected one; 0011's stale header still says "gated P3
fusion"), `B0_EXP_ID`/error-label strings, `RUN_TAG`
(`512dualv2`→`512dualv2l0`, `EXP-0011`→`EXP-0013`), the two pinned fingerprint/baseline-id
constants plus the config path inside both `python -c` gate commands, the
`EXP11_CONTRACT`→`EXP13_CONTRACT` echo labels and `levels=P2+P3`→`levels=P2-only-L0`, and
`--config` in the trainer invocation. After masking exactly those tokens and dropping
comment lines, the two bodies are byte-identical ("BODY IDENTICAL after masking allowed
tokens"): same data pins `f15cd8f5…`/`17f49430…`, same PYV snapshot-fingerprint block, same
trainer flags apart from the `--config` swap.

Fail-closed guards present in `run_exp0013_ddp4.sh` (line numbers): B0_EXP_ID gate (13),
no-args gate (22), clean-worktree gate (25), output-dir gate (28), accepted-pairs sha pin
(31), v2-split sha pin (34), PYV snapshot-fingerprint gate (37–47), architecture-fingerprint
gate (48–51), baseline-id gate (52–55). `set -Eeuo pipefail` and `B0_PREFLIGHT_ONLY`
short-circuit retained.

## 4. Code-path correctness (ue/dual.py) — OK

Static read plus live construction (`MODELS.build(Config.fromfile(...).model)`,
SAM2_CKPT env-injected):

- With `target_levels=[0]`, `levels=[0]` takes the `len(levels)==1` branch
  (ue/dual.py:99–102): `self.proj = nn.Conv2d(262, 512, 1)` and
  `self.gate = nn.Parameter(torch.zeros(512))` as plain attributes. Live-built state-dict
  fusion keys are exactly `['gate', 'proj.weight', 'proj.bias']`. EXP-0010's config uses
  scalar `target_level: 1` → same branch → same attribute names, so the EXP-0010
  checkpoint-compatibility claim holds.
- Telemetry (ue/dual.py:161–177): `single = len(self.target_levels) == 1` →
  `tag = '' if single else f'_L{t}'`; with `[0]` the recorded keys are
  `gate_mean/gate_std/gate_absmax/delta_p3_ratio/bev_coverage` with no `_L` suffix
  (`drain_telemetry` counts `gate_absmax*`-prefixed keys, suffix-agnostic).
- Freeze semantics, live-verified on the constructed model: `uav_encoder_lora=False`
  (config `uav_encoder` dict has no `use_lora` key), `requires_grad=True` params
  `0/306`, and `uav_encoder.training=False` after `model.train()`; the non-LoRA forward
  runs under `torch.no_grad()` (ue/dual.py:136–138).
- LoRA distinction is real: `use_lora: True` in this config sits in `model.backbone`
  (satellite `RSSAM2VisionEncoder`, lora_config r=16, modules_to_save=['neck']) — the
  identical key exists in the single-stream EXP-0009 config `configs/ue_whu_a3sgm_512.py`,
  so it is part of the sealed baseline shared by all four arms. The EXP-0012 UAV LoRA is a
  different key: `use_lora: True` inside the `uav_encoder` dict of
  `configs/ue_whu_a3sgm_512_dualv2_lora.py` only.

## 5. Prereg + registry consistency — OK (one provenance note)

`docs/EXP-0008_preregistration.md` §v2.3 追加 (lines 136–169) states the four-element
inheritance verbatim: primary metric = val segm/bbox AP@[.50:.95] of EXP-0013 − EXP-0009
(0010/0011 same table), selection = last epoch 100, significance = per-scene paired
percentile bootstrap with 10,000 draws seed 20261014, gates
DUAL_GAIN/DUAL_ENGAGE_NULL/DUAL_USED_BUT_NULL/DUAL_HARM with the v2.1 δ operator applied at
single-level L0 (unsuffixed telemetry keys, isomorphic to EXP-0010). The pre-registered
attribution matrix is present: 0013≈0011 → P2 single-level sufficient (simplest recipe);
0013≈0010 (no gain) → two-level synergy necessary; in-between → P2 primary + P3 secondary,
split by final-value difference. `docs/EXP_REGISTRY.md` row EXP-0013 quotes the same
identity/fingerprint and the single-variable description.

Quoted numbers:

- EXP-0010 — `grep "Epoch 100/100" /data2/wangcheng/checkpoint/surfbev_b0/launcher_logs/EXP-0010-ddp4.log`
  (line 5516, 21:54:28): `Epoch 100/100 | train=0.188565 val=disabled segm/mAP=0.7550
  bbox/mAP=0.7317 | best_segm=0.7553 best_bbox=0.7379`. Matches the v2.3 quote (0.7550/0.7317).
- EXP-0009 — the prescribed grep of
  `/data1/wangcheng/checkpoint/surfbev_b0/launcher_logs/EXP-0009-ddp4.log` **cannot
  succeed**: that file ends at Epoch 61/100 (two aborted attempts, last entry 03:44), and
  the completed run's log on `/data2/…/launcher_logs/EXP-0009-ddp4.log` was truncated to a
  124-byte output-dir guard message by the 14:18:45 chain relaunch (documented in
  `CHAIN-snapshot-retrain.log`). The numbers were instead recovered from the sealed
  checkpoint `…/EXP-0009/last_model_epoch100.pth` (`history.val_metrics` last epoch,
  100-entry history, `epoch: 99` 0-indexed): `segm/mAP = 0.759260…` → **0.7593**,
  `bbox/mAP = 0.737442…` → **0.7374**, both matching the v2.3 quote. Method
  cross-validated on EXP-0010: its checkpoint history last entry (0.755/0.7317) equals its
  surviving log's `Epoch 100/100` line exactly. Both checkpoints self-identify with the
  correct baseline ids (`ue-whu-a3sgm-512-v1`, `ue-whu-a3sgm-512-dual-v1`).

## 6. Train-ready checks — OK

- Target work dir `/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512dualv2l0_sealed_ddp4_s44_EXP-0013`
  does not exist (`ls`: No such file or directory).
- `ps aux`: the only trainer processes are EXP-0011 and EXP-0012 (4 ranks + workers each),
  with `--checkpoint-dir` pointing at their own run dirs; no process references the
  EXP-0013 path or run tag.
- Snapshot hashes recomputed now:
  `f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a  accepted_pairs.jsonl`
  and `17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5  surfbev_b0_split_v2.json`
  — identical to the wrapper pins and to the `EXP11_CONTRACT` line recorded in
  `/data2/…/launcher_logs/EXP-0011-ddp4.log` at launch (i.e., the snapshot actually used by
  0011, which is also the `--data-root` of the currently running 0011 process).
- The wrapper's PYV snapshot-fingerprint gate, re-run standalone against
  `/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen`:
  recomputed `source_payload_fingerprint_sha256 = 7b914d9aefac68a1…` == manifest value → PASS.

## Non-blocking findings

1. Launcher-log retention: the completed EXP-0009 run's stdout log was irreversibly
   truncated by a chain relaunch (`>` redirect in `chain_snapshot_and_retrain.sh` fails into
   the same file the output-dir guard refuses). The final metrics had to be recovered from
   the checkpoint's embedded per-epoch history. Recommend archiving launcher logs
   append-only (or copying at run end) before more arms launch; EXP-0011/0012 logs are
   currently live and intact.
2. Cosmetic, already-known: EXP-0011's wrapper header still carries the stale "gated P3
   fusion" wording; EXP-0013's header is the corrected template. No action required for
   this launch.

VERDICT: PASS
