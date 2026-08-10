#!/usr/bin/env python
"""多卡通信连通性自检 —— 在真训练之前先确认进程组和集合通信本身能用。

**为什么单独一个脚本**：多卡训练跑不起来时，「通信层坏了」与「训练逻辑坏了」的
现象很像（都表现为卡住或莫名退出），但排查方向完全不同。先用几秒钟把通信层验干净，
后面出问题就能直接排除这一层。

海光 DCU 上尤其值得先跑：DTK 的通信后端是 RCCL（AMD 对 NCCL 的实现），PyTorch 侧
后端名仍叫 ``"nccl"``，但底层库是否真的装全、能否跨卡建链，只有实测知道。

用法（在项目根目录）：

    python -m torch.distributed.run --nnodes 1 --nproc_per_node 2 tools/check_ddp_comm.py

``--nproc_per_node`` 填要测的卡数。注意**必须**用 torchrun 起（本脚本依赖它注入
RANK / LOCAL_RANK / WORLD_SIZE）；直接 ``python tools/check_ddp_comm.py`` 会走单进程
分支、只做一次自检就退出，也是有意义的（验证单卡路径不受影响）。

检查项按依赖顺序排，前一项失败后面就不必看：
  1. 环境变量与 torch build（后端名、可见设备数）
  2. 绑卡（set_device）
  3. 建进程组（init_process_group）—— DCU 上最可能卡住的一步
  4. all_reduce 数值正确性 —— 通了但结果错说明拓扑或数据类型有问题
  5. broadcast —— DDP 构造时同步初始参数用的就是它
  6. barrier —— rank0 落盘后其余 rank 等待用的
  7. bf16 集合通信 —— 训练实际用的 dtype，某些老 RCCL 只支持 fp32/fp16
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 让 utils.* 可 import（本脚本在 tools/ 下，仓库根在上一级）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _say(rank: int, msg: str) -> None:
    """带 rank 前缀输出。所有 rank 都打 —— 本脚本的价值就在于看到**每个** rank
    的状态，只看 rank 0 会漏掉「某张卡坏了」这种最需要发现的情况。"""
    print(f"[rank {rank}] {msg}", flush=True)


def main() -> int:
    # torch 缺失时给一句人话而不是裸 traceback —— 本脚本常被复制到别的机器 /
    # 别的 venv 里跑，「跑错解释器」是最常见的失败原因，而 ModuleNotFoundError
    # 的 traceback 并不会提示这一点。
    try:
        import torch
    except ImportError:
        print("未找到 torch —— 当前解释器不是训练用的那个环境。", flush=True)
        print(f"  当前: {sys.executable}", flush=True)
        print("  DTK 镜像里通常是 /usr/local/bin/python；若项目建了 venv 则用 venv/bin/python。",
              flush=True)
        return 1

    from utils import accelerator, distributed as dist_env

    rank = dist_env.rank()
    world = dist_env.world_size()
    local = dist_env.local_rank()

    # ── 1. 环境与 build ────────────────────────────────────────────────
    info = accelerator.detect()
    if rank == 0:
        print("=" * 66, flush=True)
        print(" 多卡通信自检", flush=True)
        print("=" * 66, flush=True)
        print(f"torch          : {info.torch_version}", flush=True)
        print(f"后端           : {info.vendor_label}"
              f"（cuda={info.cuda_version} hip={info.hip_version}）", flush=True)
        print(f"通信后端       : {dist_env.backend_name()}"
              f"{'  ← DTK 上底层是 RCCL' if info.backend == 'dcu' else ''}", flush=True)
        print(f"可见设备数     : {torch.cuda.device_count()}", flush=True)
        print(f"拓扑           : {dist_env.topology_summary()}", flush=True)
        print("-" * 66, flush=True)

    if not dist_env.is_distributed():
        _say(rank, "WORLD_SIZE<=1 —— 没在 torchrun 下（或 --nproc_per_node=1）。")
        _say(rank, "单卡路径正常，但通信未被测试。要测多卡请用：")
        _say(rank, "  python -m torch.distributed.run --nnodes 1 "
                   "--nproc_per_node 2 tools/check_ddp_comm.py")
        return 0

    if not torch.cuda.is_available():
        _say(rank, "FAIL: torch.cuda.is_available() = False。"
                   "容器是否挂了 /dev/kfd 与 /dev/dri？")
        return 1

    # ── 2-3. 绑卡 + 建进程组 ───────────────────────────────────────────
    # init() 内部按 local_rank 绑卡再建组（顺序很关键，见该函数 docstring）。
    # DCU 上如果 RCCL 缺库 / 卡间不通，通常就卡死或抛在这一步。
    try:
        dist_env.init()
    except Exception as exc:  # noqa: BLE001
        _say(rank, f"FAIL 建进程组: {type(exc).__name__}: {exc}")
        return 1
    dev = f"cuda:{local}"
    _say(rank, f"进程组就绪，绑定 {dev}（{torch.cuda.get_device_name(local)}）")

    import torch.distributed as dist

    failures: list[str] = []

    def _check(name: str, fn) -> None:
        try:
            fn()
            _say(rank, f"  [ok]   {name}")
        except Exception as exc:  # noqa: BLE001
            _say(rank, f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
            failures.append(name)

    # ── 4. all_reduce 数值正确性 ───────────────────────────────────────
    # 每个 rank 贡献 (rank+1)，SUM 后应等于 1+2+...+world = world*(world+1)/2。
    # 只验「不抛」不够：拓扑配错时可能算出各 rank 不一致的结果，必须对数值。
    expected = world * (world + 1) / 2

    def _all_reduce_sum():
        t = torch.full((1,), float(rank + 1), device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        got = t.item()
        if abs(got - expected) > 1e-6:
            raise AssertionError(f"期望 {expected}，实得 {got}")

    _check(f"all_reduce SUM（期望 {expected:g}）", _all_reduce_sum)

    # ── 5. broadcast ───────────────────────────────────────────────────
    # DDP 构造时用它把 rank0 的初始参数同步给其他 rank（_sync_module_states）。
    def _broadcast():
        t = torch.full((4,), float(rank), device=dev)
        dist.broadcast(t, src=0)
        if abs(t[0].item()) > 1e-6:
            raise AssertionError(f"应被 rank0 的 0.0 覆盖，实得 {t[0].item()}")

    _check("broadcast（rank0 → 全体）", _broadcast)

    # ── 6. barrier ─────────────────────────────────────────────────────
    _check("barrier", lambda: dist.barrier())

    # ── 7. bf16 集合通信 ───────────────────────────────────────────────
    # 训练实际用 bf16。某些较老的 RCCL 只对 fp32/fp16 有 kernel，bf16 会在这里挂 ——
    # 而那时的报错发生在训练第一步的梯度同步里，不容易联想到是 dtype 问题。
    def _all_reduce_bf16():
        t = torch.full((1024,), float(rank + 1), device=dev, dtype=torch.bfloat16)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        got = t[0].float().item()
        # bf16 只有 8 位有效位，用相对误差
        if abs(got - expected) / expected > 0.02:
            raise AssertionError(f"期望约 {expected}，实得 {got}")

    _check("all_reduce bf16（训练实际 dtype）", _all_reduce_bf16)

    # ── 汇总 ───────────────────────────────────────────────────────────
    dist.barrier()
    if rank == 0:
        print("-" * 66, flush=True)
        if failures:
            print(f"结论：{len(failures)} 项失败 —— {', '.join(failures)}", flush=True)
            print("多卡训练在修好这些之前不会正常工作。", flush=True)
        else:
            print("结论：通信层全部正常，可以开多卡训练。", flush=True)
        print("=" * 66, flush=True)

    dist_env.destroy()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
