# WHU / SurfaceBEV migration

Source: `95026ae677dd0a8d77a3ee4c5fb68abeb4ea01c1`.
Original committed baseline: `27b42d1`.
Archive branch: `codex/archive-dual-stream-pre-whu-20261013`.
Migration branch: `codex/whu-surfacebev-migration`.

Recovery snapshot: `/home/wangcheng/project/new/migration-backups/whu-surfacebev-20261013`.
It contains a binary HEAD diff, full untracked-file archive (symlinks preserved),
status and content-hash inventory. Restore into a fresh worktree at the original
HEAD using `git apply tracked.patch`, then extract `untracked.tar` there.
The original worktree, datasets and experiment outputs are not migration targets.

Only mainline source/protocol changes are adopted; experiment data/output files
remain external and their inventory is in the recovery manifest. Historical
protocols retain their original experimental identity. Old P2/P3 and A1 routes
are archived; new end-to-end fusion and full training are outside this change.
