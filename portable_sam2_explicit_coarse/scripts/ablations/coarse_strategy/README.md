# Coarse strategy screening

These wrappers keep the C2 architecture fixed (PAFPN + adaptive coarse 2P2N,
without box, dense prompt, or DenseBR) and isolate the coarse-loss weighting:

| Script | Coarse loss |
|---|---|
| `s0_c2_fixed_w010.sh` | fixed 0.10 |
| `s1_c2_fixed_w020.sh` | fixed 0.20 |
| `s2_c2_two_stage_w020_w010.sh` | epochs 1-5: 0.20; epoch 6+: 0.10 |

They inherit the shared ablation dataset defaults and enable prompt diagnostics
every 50 training iterations. Override `PROMPT_DEBUG_STATS_INTERVAL` when a
denser trace is required.
