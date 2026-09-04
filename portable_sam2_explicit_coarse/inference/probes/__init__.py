"""Archived audit probes (pathway audit, oracle, E0, learnability, P2 visualization).

Run directly, e.g.
  python inference/probes/probe_decoder_pathways.py --checkpoint <ckpt>
They are one-shot audit tooling kept for evidence reproduction; the P2 pathway
audit conclusions they produced live in docs/p2_decoder_side_findings.md and
commit a94a31c. Note: scripts/ablate_whu_inference.sh (the batch launcher) was
removed with the historical ablation generation — invoke probes manually.
"""
