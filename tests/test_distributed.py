"""utils/distributed.py —— DDP 判定与集合通信的单元测试。

本模块是三个并行改动（sampler 分片 / 训练循环 / torchrun 启动器）的共同契约，
所以这里的断言比一般单测更严：**单进程行为一旦变化，三处都会静默出错**。

不真起多进程。torchrun 的行为通过设环境变量模拟 —— 这正是被测模块的输入来源
（它只读 RANK / LOCAL_RANK / WORLD_SIZE，不自己拉进程），所以模拟是充分的。
"""
from __future__ import annotations

import sys
import types

import pytest

sys.path.insert(0, ".")

from utils import distributed as d  # noqa: E402

_ENV_KEYS = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """每个用例都从「没在 torchrun 下」起步，并复位模块级初始化标记。

    torchrun 的变量若从真实环境泄漏进来（CI runner 上有可能），单进程用例会
    静默变成多进程用例 —— 那种失败极难定位，所以显式清干净。
    """
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    d._INITIALIZED = False
    yield
    d._INITIALIZED = False


def _torchrun(monkeypatch, *, rank: int, world: int, local: int | None = None):
    """模拟 torchrun 注入的环境。"""
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", str(world))
    monkeypatch.setenv("LOCAL_RANK", str(local if local is not None else rank))


# ---------------------------------------------------------------------------
# 单进程：必须逐字节保持原行为
# ---------------------------------------------------------------------------


def test_single_process_defaults():
    """没有 torchrun 时的全套默认值。

    这组值是三个下游改动的地基：`is_distributed()` 为 False 让所有新分支失效，
    `is_main()` 为 True 让所有 rank0 门控恒过。任一变化都会让「不开多卡的用户
    行为不变」这条硬约束破掉。
    """
    assert d.world_size() == 1
    assert d.rank() == 0
    assert d.local_rank() == 0
    assert d.is_distributed() is False
    assert d.is_main() is True
    assert d.topology_summary() == "单卡"


def test_single_process_init_is_noop():
    """`init()` 在单进程下不建进程组、返回 False —— 调用方可以无条件调它。"""
    assert d.init() is False
    assert d._INITIALIZED is False


def test_destroy_is_idempotent_without_init():
    """没 init 过就 destroy 不该抛 —— 异常退出路径会走到这里。"""
    d.destroy()
    d.destroy()


def test_barrier_and_reduce_are_noop_single_process():
    d.barrier()  # 不抛即通过
    assert d.all_reduce_mean(3.5) == 3.5
    assert d.all_reduce_mean(0) == 0


# ---------------------------------------------------------------------------
# 多进程判定
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rank,world,local", [(0, 2, 0), (1, 2, 1), (3, 8, 3)])
def test_reads_torchrun_env(monkeypatch, rank, world, local):
    _torchrun(monkeypatch, rank=rank, world=world, local=local)
    assert d.rank() == rank
    assert d.world_size() == world
    assert d.local_rank() == local
    assert d.is_distributed() is True
    assert d.is_main() is (rank == 0)


def test_nproc_1_treated_as_single_process(monkeypatch):
    """`torchrun --nproc_per_node=1` 会设全套变量但只有一个进程。

    判据刻意用 `world_size() > 1` 而非「变量是否存在」：那种情况语义上等价于单卡，
    按单进程处理可以省掉建进程组和包 DDP 的开销，且行为完全一致。
    """
    _torchrun(monkeypatch, rank=0, world=1)
    assert d.is_distributed() is False
    assert d.is_main() is True
    assert d.init() is False


def test_local_rank_differs_from_rank_multinode(monkeypatch):
    """多机时 local_rank 与 rank 不同 —— 绑卡必须用 local_rank。

    2 机 × 2 卡：rank 2 在第二台机器上是 local_rank 0。用 rank 绑卡会试图访问
    不存在的 2 号卡（每台机器只有 2 张），或者更糟：绑到别的 rank 正在用的卡上。
    """
    _torchrun(monkeypatch, rank=2, world=4, local=0)
    assert d.rank() == 2
    assert d.local_rank() == 0
    assert d.is_main() is False


def test_topology_summary_mentions_ranks(monkeypatch):
    _torchrun(monkeypatch, rank=1, world=4)
    s = d.topology_summary()
    assert "4" in s and "1" in s


# ---------------------------------------------------------------------------
# 环境变量非法值：一律退回单进程，不能抛
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["abc", "", "1.5", "-1"])
def test_malformed_world_size_falls_back_to_single(monkeypatch, bad):
    """非法 WORLD_SIZE 退回 1 而不是抛。

    这些查询函数在训练启动最早期就被调用（决定 device、打日志），此时抛异常会让
    用户看到一个与真实问题无关的 traceback。退回单进程是安全的降级：最坏情况是
    多卡没生效，而不是训练起不来。
    """
    monkeypatch.setenv("WORLD_SIZE", bad)
    assert d.world_size() == 1
    assert d.is_distributed() is False


@pytest.mark.parametrize("bad", ["abc", "", "-3"])
def test_malformed_rank_falls_back_to_zero(monkeypatch, bad):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", bad)
    assert d.rank() == 0
    assert d.is_main() is True


# ---------------------------------------------------------------------------
# 通信后端选择
# ---------------------------------------------------------------------------


def _fake_torch(monkeypatch, *, cuda_available: bool):
    mod = types.ModuleType("torch")
    mod.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: 2,
        set_device=lambda i: None,
    )
    monkeypatch.setitem(sys.modules, "torch", mod)
    return mod


def test_backend_is_nccl_on_gpu(monkeypatch):
    """GPU 上一律 "nccl" —— 海光 DTK 也是。

    ROCm / DTK build 把 NCCL 符号映射到 RCCL，后端名不变。传 "rccl" 会 ValueError。
    这条钉住的是「不要为 DCU 特殊处理后端名」这个反直觉的事实。
    """
    _fake_torch(monkeypatch, cuda_available=True)
    assert d.backend_name() == "nccl"


def test_backend_falls_back_to_gloo_without_gpu(monkeypatch):
    _fake_torch(monkeypatch, cuda_available=False)
    assert d.backend_name() == "gloo"


def test_backend_survives_torch_import_failure(monkeypatch):
    """torch import 挂了也要给出个后端名，不能抛。"""
    monkeypatch.setitem(sys.modules, "torch", None)
    assert d.backend_name() in ("nccl", "gloo")


# ---------------------------------------------------------------------------
# init()：绑卡顺序与越界检查
# ---------------------------------------------------------------------------


def test_init_binds_device_before_creating_process_group(monkeypatch):
    """`set_device` 必须在 `init_process_group` **之前**调用。

    NCCL 用当前设备决定通信拓扑。不绑就建组的话所有 rank 都会往 0 号卡挤，表现是
    显存爆在一张卡上 + 通信死锁 —— 多卡最经典的坑。这条测试记录调用顺序，防止
    以后重构时被调换。
    """
    calls: list[str] = []
    torch = _fake_torch(monkeypatch, cuda_available=True)
    torch.cuda.set_device = lambda i: calls.append(f"set_device({i})")

    dist = types.ModuleType("torch.distributed")
    dist.is_initialized = lambda: False
    dist.init_process_group = lambda **kw: calls.append(f"init_process_group({kw['backend']})")
    dist.destroy_process_group = lambda: calls.append("destroy")
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    torch.distributed = dist

    _torchrun(monkeypatch, rank=1, world=2, local=1)
    assert d.init() is True
    assert calls == ["set_device(1)", "init_process_group(nccl)"], calls


def test_init_rejects_local_rank_beyond_device_count(monkeypatch):
    """请求的进程数超过实际卡数 → 明确报错，且信息要能自助排查。

    最常见的原因是 --nproc_per_node 写大了，或 CUDA_VISIBLE_DEVICES /
    HIP_VISIBLE_DEVICES 限制了可见范围。不检查的话 set_device 会抛一个不提这两点
    的底层错误，用户很难定位。
    """
    torch = _fake_torch(monkeypatch, cuda_available=True)
    torch.cuda.device_count = lambda: 2
    dist = types.ModuleType("torch.distributed")
    dist.is_initialized = lambda: False
    dist.init_process_group = lambda **kw: None
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    torch.distributed = dist

    _torchrun(monkeypatch, rank=3, world=4, local=3)
    with pytest.raises(RuntimeError) as ei:
        d.init()
    msg = str(ei.value)
    assert "LOCAL_RANK=3" in msg
    assert "2" in msg
    assert "VISIBLE_DEVICES" in msg


def test_init_reuses_externally_created_group(monkeypatch):
    """进程组已被外部建好时复用它，且**不接管销毁责任**。

    `_INITIALIZED` 保持 False，这样 `destroy()` 不会去拆别人建的组 —— 拆了会让
    外层框架后续的集合通信全部失败。
    """
    torch = _fake_torch(monkeypatch, cuda_available=True)
    dist = types.ModuleType("torch.distributed")
    dist.is_initialized = lambda: True
    dist.init_process_group = lambda **kw: pytest.fail("不该重复建进程组")
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    torch.distributed = dist

    _torchrun(monkeypatch, rank=0, world=2)
    assert d.init() is True
    assert d._INITIALIZED is False


def test_destroy_only_cleans_own_group(monkeypatch):
    torch = _fake_torch(monkeypatch, cuda_available=True)
    destroyed: list[bool] = []
    dist = types.ModuleType("torch.distributed")
    dist.is_initialized = lambda: True
    dist.destroy_process_group = lambda: destroyed.append(True)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    torch.distributed = dist

    d._INITIALIZED = True
    d.destroy()
    assert destroyed == [True]
    assert d._INITIALIZED is False
    # 再调一次不该重复销毁
    d.destroy()
    assert destroyed == [True]


def test_destroy_swallows_errors(monkeypatch):
    """销毁失败不该盖掉真正的训练异常 —— 它常在 finally 里被调用。"""
    torch = _fake_torch(monkeypatch, cuda_available=True)
    dist = types.ModuleType("torch.distributed")
    dist.is_initialized = lambda: True

    def _boom():
        raise RuntimeError("NCCL 通信器已失效")

    dist.destroy_process_group = _boom
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    torch.distributed = dist

    d._INITIALIZED = True
    d.destroy()  # 不抛即通过
    assert d._INITIALIZED is False
