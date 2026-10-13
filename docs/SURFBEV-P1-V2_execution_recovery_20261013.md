# P1 执行托管恢复（2026-10-13）

此前 evaluate 进程退出，日志无异常、未记录退出码；最后检查时 CPU 正在计算、RSS
约 1GB、系统内存充足，不能确认终止原因。前台启动会话生命周期是待核实原因，
不将其写成已证实的故障结论。

已核验 train=694、val=100、遥测=794；五个缓存实现 SHA 与当前源文件逐一一致。
test lock 不存在，test 分片=0，未开封测试集。

恢复只新增外层 `scripts/ue/run_p1_supervised.py`。已审计 P1 runner 与五个依赖文件
均不改动，缓存合同仍严格检查。supervisor 直接子进程运行原 runner、透传参数，
不导入或替换任何 P1 函数；每10秒原子写心跳与test缓存进度，退出后写 returncode。
启动以 `subprocess.Popen(start_new_session=True)` 建立独立会话，stdin 为 DEVNULL，
stdout/stderr 直接写数据盘日志。避免依赖交互会话和 tee 管道保持生命周期。

工件根目录：
`/data1/wangcheng/checkpoint/surfbev_b0/probes/SURFBEV-P1-V2-R1a-20261012/`

恢复日志：`evaluation_supervised_20261013.log`。
运行状态：`evaluation_runtime.json`（含 supervisor/worker PID、状态、退出码和时间）。
本记录不改变三个实验臂、val选择、test锁、指标、bootstrap或P1门，不构成效果结论。

## 启动证据

独立复审 R1c PASS 已追加到 `docs/reviews/SURFBEV-P1-V2-R1_recovery_audit.md`。
启动器无数据 smoke（仅 runner --help）成功记录子进程正常退出 code=0。
正式恢复于 2026-10-13 11:18:40 +08:00 启动：supervisor PID=1666307，worker
PID=1666309。启动后 ps 确认 supervisor PPID=1、SID=1666307。
外层以 Popen(start_new_session=True, stdin=DEVNULL, stdout=上述日志文件,
stderr=STDOUT) 启动；使用 cvt2 实际解释器 `/data2/wangcheng/envs/cvt2/bin/python`。
参数为 `--phase evaluate --device cuda:0 --allow-test`，output-dir 为原 R1a 工件根目录。
当时 runtime.state=running、test_lock_exists=false、test_scenes_cached=0。
