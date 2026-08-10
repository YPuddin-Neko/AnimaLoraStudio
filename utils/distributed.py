"""分布式训练（DDP）的单一权威源 —— rank 查询、进程组生命周期、集合通信。

设计原则与 :mod:`utils.accelerator` 一致：**判定收敛在这一处**，上层只消费结论，
不自己读 ``os.environ["RANK"]`` 或调 ``dist.get_rank()``。加第二种并行方式
（FSDP / pipeline）时只改这个文件。

单进程是默认且必须逐字节保持原行为
------------------------------------
没有 torchrun 时 ``is_distributed()`` 为 False、``rank()`` 为 0、
``world_size()`` 为 1，所有 ``is_main()`` 门控恒真。也就是说**不开多卡的用户
（含所有 NVIDIA 单卡用户）走的代码路径与改动前完全一致** —— 这是本模块所有 API
的设计约束，不是巧合。

启动方式
--------
``torchrun --nproc_per_node=N runtime/anima_train.py --config ...``。torchrun 会给
每个子进程注入 ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` / ``MASTER_ADDR`` /
``MASTER_PORT``。本模块只读这些环境变量，不自己拉进程 —— 谁启动的由
``studio/supervisor/cmd_builder.py`` 决定。

海光 DCU 上的通信后端
--------------------
仍然传 ``"nccl"``。ROCm / DTK build 的 PyTorch 把 NCCL 符号映射到 RCCL
（AMD 的等价实现），后端名不变 —— 与 ``torch.cuda.*`` 在 HIP 上复用 CUDA 命名
同理。传 ``"rccl"`` 会 ValueError。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: 进程组是否由本模块初始化过（``destroy()`` 用来决定要不要清理）。
_INITIALIZED = False


def _env_int(name: str) -> Optional[int]:
    """读环境变量里的整数；缺失 / 非法一律 None（视为「没在 torchrun 下」）。"""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("环境变量 %s=%r 不是整数，按单进程处理", name, raw)
        return None


def world_size() -> int:
    """参与训练的进程总数。单进程（无 torchrun）为 1。

    优先读环境变量而不是 ``dist.get_world_size()``：本函数在进程组初始化**之前**
    就要能用（``cmd_builder`` 决定 batch size 语义、日志打印拓扑都在那之前）。
    """
    ws = _env_int("WORLD_SIZE")
    return ws if ws and ws > 0 else 1


def rank() -> int:
    """本进程的全局序号（0 .. world_size-1）。单进程为 0。"""
    r = _env_int("RANK")
    return r if r is not None and r >= 0 else 0


def local_rank() -> int:
    """本进程在**本机**的序号 —— 决定绑哪张卡。单进程为 0。

    单机多卡时与 :func:`rank` 相同；多机时不同（每台机器的 local_rank 都从 0 数）。
    绑卡一律用这个，不要用 ``rank()``。
    """
    lr = _env_int("LOCAL_RANK")
    return lr if lr is not None and lr >= 0 else 0


def is_distributed() -> bool:
    """是否在多进程 DDP 下运行。

    判据是 ``world_size() > 1`` 而非「环境变量存在」：``torchrun
    --nproc_per_node=1`` 会设全套变量但只有一个进程，那种情况按单进程处理更省
    （不必初始化进程组、不必包 DDP），且语义上确实等价。
    """
    return world_size() > 1


def is_main() -> bool:
    """本进程是否为 rank 0 —— checkpoint / 采样 / 日志 / 进度上报的门控。

    单进程恒为 True，所以 ``if is_main():`` 包住的代码在不开多卡时全部照常执行。
    """
    return rank() == 0


def backend_name() -> str:
    """当前该用的通信后端名。

    GPU 上一律 ``"nccl"`` —— 海光 DTK / ROCm build 把它映射到 RCCL，后端名不变
    （传 ``"rccl"`` 会 ValueError）。没有 GPU 时用 ``"gloo"``（CPU 通信，仅用于
    调试拓扑，真训练不会走）。
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "nccl"
    except Exception:  # noqa: BLE001
        pass
    return "gloo"


def init() -> bool:
    """初始化进程组并把本进程绑到 ``local_rank`` 对应的卡。返回是否真的初始化了。

    单进程下**什么都不做**并返回 False —— 调用方可以无条件调它，不必自己判断。

    绑卡（``torch.cuda.set_device``）必须在建进程组**之前**：NCCL 用当前设备决定
    通信拓扑，不绑的话所有 rank 都会往 0 号卡挤，表现是显存爆在一张卡上 + 通信
    死锁。这是多卡最经典的坑，所以放在这里做掉，不留给调用方。
    """
    global _INITIALIZED
    if not is_distributed():
        return False
    if _INITIALIZED:
        return True

    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        # 别人（外部框架 / 测试）已经建好了，认它，但不接管销毁责任
        logger.info("进程组已由外部初始化，直接复用")
        return True

    lr = local_rank()
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if lr >= count:
            raise RuntimeError(
                f"LOCAL_RANK={lr} 超出本机可见设备数 {count}。"
                f"检查 --nproc_per_node 是否大于实际卡数，"
                f"或 CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES 是否限制了可见范围。"
            )
        torch.cuda.set_device(lr)

    dist.init_process_group(backend=backend_name())
    _INITIALIZED = True
    logger.info(
        "DDP 就绪：rank %d/%d（local_rank %d，backend %s）",
        rank(), world_size(), lr, backend_name(),
    )
    return True


def destroy() -> None:
    """销毁进程组。只清理本模块建的那个，幂等，失败不抛。

    训练正常结束与异常退出都要调 —— 不销毁会让 NCCL 留下僵死通信器，下一个训练
    任务起来时可能卡在 init_process_group 上（同一张卡上的旧通信器没释放）。
    """
    global _INITIALIZED
    if not _INITIALIZED:
        return
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001  清理失败不该盖掉真正的训练异常
        logger.debug("销毁进程组失败: %s", exc)
    finally:
        _INITIALIZED = False


def barrier() -> None:
    """等所有 rank 到齐。单进程下 no-op。

    用在「rank 0 做完某件事其他 rank 才能继续」的地方（如 rank 0 写完 checkpoint
    再一起进下一 epoch）。**不要**用它来同步数据，那用 :func:`all_reduce_mean`。
    """
    if not is_distributed():
        return
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
    except Exception as exc:  # noqa: BLE001
        logger.debug("barrier 失败: %s", exc)


def all_reduce_mean(value: Any) -> Any:
    """把标量 / 张量在所有 rank 间取平均，返回平均值。单进程下原值返回。

    给 loss 日志用：各 rank 只见到自己那份 micro-batch 的 loss，直接打出来会让
    曲线抖且与单卡不可比。平均后语义等于「全局 batch 的 loss」。

    入参可以是 float 或 0-d/1-d tensor；float 会临时搬到当前卡上做通信。
    失败时返回原值（日志不该阻塞训练）。
    """
    if not is_distributed():
        return value
    try:
        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            return value
        is_scalar = not isinstance(value, torch.Tensor)
        t = torch.tensor(
            float(value), device=f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu",
        ) if is_scalar else value.detach().clone().float()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= world_size()
        return t.item() if is_scalar else t
    except Exception as exc:  # noqa: BLE001
        logger.debug("all_reduce_mean 失败，返回本 rank 原值: %s", exc)
        return value


def topology_summary() -> str:
    """一行拓扑描述，给启动日志用。单进程返回 "单卡"。"""
    if not is_distributed():
        return "单卡"
    return f"DDP {world_size()} 进程（本进程 rank {rank()} / local_rank {local_rank()}）"
