"""batch sampler 的 DDP 分片（ADR 0019 多卡）—— 单进程零回归 + 各 rank 严格等长。

守两类东西：
1. **单进程逐字节不变**。所有不开多卡的用户（含全部 NVIDIA 单卡用户）走这条路，
   ``__iter__`` / ``__len__`` 必须与加 DDP 支持之前完全一致 —— 这里断言具体的
   batch 列表（不是"数量对就行"），改动 sampler 时一眼能看出语义漂移。
2. **分片不变式**。各 rank batch 数严格相等（不等 → 少的那个 rank 先退出循环，
   其余 rank 永远等在 all_reduce 上，表现为训练静默卡死）；并集 ⊆ 单进程集合
   （不能凭空造 batch）；rank 间无重叠；每个 batch 内同桶（跨桶 → collate 的
   torch.stack 形状不一致直接 RuntimeError）。

多进程用 monkeypatch 改 ``utils.distributed`` 的 world_size / rank 模拟，不真起进程：
被测的是纯 Python 的划分逻辑，起 torchrun 只会把测试变慢变脆。
"""
from __future__ import annotations

import importlib.util
import random
import sys
import types

import pytest

from utils import distributed as dist_env

# ---------------------------------------------------------------------------
# torch 缺失时的 import 兜底
#
# runtime/training/dataset.py 顶部 `import torch` + `class ImageDataset(Dataset)`，
# 且它 import 的 .families 链上还有 `from torch import Tensor` 之类 —— 而 sampler
# 逻辑本身是纯 Python。开发机 / 无 GPU CI 上没有 torch 时，直接
# `from training.dataset import ...` 会 collection error（既有
# tests/test_bucket_batch_sampler.py / test_navit_pack_sampler.py 就是这样跑不起来的）。
# 这里挂一个按需生成属性的 torch 桩（_StubTorchFinder）让 import 期过去，**并在
# import 完成后把 sys.modules 恢复原状** —— 桩留在 sys.modules 里会让后续 test 模块
# 拿到假 torch（比 collection error 难查得多）。装了真 torch 时下面整段是 no-op。
# ---------------------------------------------------------------------------


class _StubAttr:
    """任意属性 / 任意调用都能过的占位对象（只服务 import 期，不参与计算）。"""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, key):
        if key.startswith("__"):
            raise AttributeError(key)
        val = _stub_value(f"{self._name}.{key}", key)
        setattr(self, key, val)
        return val

    def __call__(self, *a, **kw):
        # `@torch.no_grad()` 这类装饰器：调用要返回可再调用的东西
        return _StubAttr(f"{self._name}()")


def _stub_value(qualname, leaf):
    """首字母大写 → 空类（``from torch import Tensor`` 后常当基类/注解用）；否则 → _StubAttr。"""
    if leaf[:1].isupper():
        return type(leaf, (), {})
    return _StubAttr(qualname)


class _StubModule(types.ModuleType):
    def __getattr__(self, key):
        if key == "__version__":
            return "0.0.0+stub"  # 有模块会 parse 版本号比较能力门控
        if key.startswith("__"):
            raise AttributeError(key)
        val = _stub_value(f"{self.__name__}.{key}", key)
        setattr(self, key, val)
        return val


class _StubTorchFinder:
    """让任意 ``torch`` / ``torch.*`` import 都解析到 _StubModule。

    只挂 finder 不够 —— ``import torch.nn.functional`` 走的是 sys.meta_path 查找，
    光往 sys.modules 塞几个假模块盖不住；反过来只有 finder 也够不着
    ``from torch import Tensor``（那是属性访问），所以 _StubModule 还要自动生属性。
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "torch" and not fullname.startswith("torch."):
            return None
        # is_package=True → 桩模块带 __path__，否则 `from torch.utils.data import ...`
        # 会在 "'torch' is not a package" 上失败。
        return importlib.util.spec_from_loader(fullname, self, is_package=True)

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


def _load_dataset_module():
    """import training.dataset，torch 缺失时用桩兜底并在事后恢复 sys.modules。"""
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001  Windows 上 torch 坏了是 OSError 而非 ImportError
        pass
    else:
        import training.dataset as mod
        return mod

    finder = _StubTorchFinder()
    sys.meta_path.insert(0, finder)
    before = set(sys.modules)
    try:
        import training.dataset as mod
        return mod
    finally:
        sys.meta_path.remove(finder)
        # 桩 torch + 用桩 import 出来的 training.* 都从 sys.modules 撤掉：留着会让后续
        # test 模块拿到假 torch（比 collection error 难查得多）。本模块已经抓住了
        # 需要的类对象，撤掉不影响自己。
        # 反序（子模块先于父包）+ 顺手删父包上的属性：只 pop sys.modules 的话，
        # `training` 包对象上还留着 `training.dataset` 属性，后续 `from training import
        # dataset` 会拿到桩版本（sys.modules 那条查找根本不走）。
        for name in sorted(set(sys.modules) - before, reverse=True):
            if not (name == "torch" or name.startswith(("torch.", "training."))):
                continue
            sys.modules.pop(name, None)
            parent, _, leaf = name.rpartition(".")
            pmod = sys.modules.get(parent) if parent else None
            if pmod is not None and hasattr(pmod, leaf):
                delattr(pmod, leaf)


_ds = _load_dataset_module()
BucketBatchSampler = _ds.BucketBatchSampler
NavitPackBatchSampler = _ds.NavitPackBatchSampler


# ------------------------------------------------------------------ fixtures


class _MockBucketedDataset:
    """模拟 CachedLatentDataset：暴露 bucket_for_index 走 BucketBatchSampler 的桶分支。"""

    def __init__(self, bucket_for_index):
        self.bucket_for_index = list(bucket_for_index)

    def __len__(self):
        return len(self.bucket_for_index)

    def __getitem__(self, idx):
        return idx


class _PlainDataset:
    """无桶信息 → BucketBatchSampler 退回线性分批（也要能分片）。"""

    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return idx


class _RepeatWrapper:
    """模拟 RepeatDataset(CachedLatentDataset)：外层更长、桶信息在内层。

    这是 Kohya 风格 `5_concept` 目录的真实形态，走 sampler 里的 ``idx % base_len``
    分支 —— 分桶 + 分片都要在这条路上成立。
    """

    def __init__(self, inner, repeats):
        self.dataset = inner
        self._repeats = int(repeats)

    def __len__(self):
        return len(self.dataset) * self._repeats

    def __getitem__(self, idx):
        return idx % len(self.dataset)


class _FakeTokenDataset:
    """模拟 NaViT 路径：token_count_for_index 直给。"""

    def __init__(self, token_counts):
        self.token_count_for_index = list(token_counts)

    def __len__(self):
        return len(self.token_count_for_index)


def _fake_topology(monkeypatch, world_size, rank):
    """把 utils.distributed 的拓扑改成 (world_size, rank)。

    改的是 utils.distributed 上的属性而非 training.dataset 的全局名 —— dataset.py
    有意 `from utils import distributed as dist_env` 拿模块对象、调用时才查属性，
    正是为了让这种 patch 打得中（也顺便钉住那个 import 形式不能改成 from-import）。
    """
    monkeypatch.setattr(dist_env, "world_size", lambda: world_size)
    monkeypatch.setattr(dist_env, "rank", lambda: rank)
    monkeypatch.setattr(dist_env, "is_distributed", lambda: world_size > 1)


def _bucketed(sizes):
    """按 {桶: 样本数} 生成 bucket_for_index（桶 key 用 (i, i) 便于辨识）。"""
    out = []
    for i, n in enumerate(sizes):
        out += [(i, i)] * n
    return out


def _shards(make_sampler, monkeypatch, world_size, epoch=None):
    """在给定 world_size 下逐 rank 跑一遍，返回 [(batches, len(sampler)), ...]。

    每个 rank 都新建 sampler：真 DDP 下就是 world_size 个互不相干的进程各建一个，
    共用一个实例会让 NavitPackBatchSampler 的 _cached_packs 串味。
    """
    result = []
    for r in range(world_size):
        with monkeypatch.context() as m:
            _fake_topology(m, world_size, r)
            sampler = make_sampler()
            if epoch is not None:
                sampler.set_epoch(epoch)
            batches = [list(b) for b in sampler]
            result.append((batches, len(sampler)))
    return result


def _single(make_sampler, monkeypatch, epoch=None):
    """单进程（world_size=1）下的完整 batch 列表 —— 分片测试的比较基线。"""
    (batches, _n), = _shards(make_sampler, monkeypatch, 1, epoch=epoch)
    return batches


def _bucket_of(dataset, batch):
    """返回 batch 内所有样本的桶（顺带断言同桶）。"""
    lst = _ref_cached(dataset).bucket_for_index   # 走 RepeatDataset 包装时要取内层
    buckets = {tuple(lst[i % len(lst)]) for i in batch}
    assert len(buckets) == 1, f"batch {batch} 跨桶：{buckets}（collate 的 torch.stack 会炸）"
    return buckets.pop()


# --------------------------------------------------- 改动前实现的逐字节参照实现
# 下面两个函数是 BucketBatchSampler.__iter__ / __len__ 在**加 DDP 支持之前**的
# 实现原样拷贝。用参照实现比对而不是硬编码 shuffle 结果，是因为后者依赖
# random.Random 的内部算法（跨 Python 版本理论上可变），而这里真正要钉的是
# 「单进程语义没漂」。硬编码的那份见 test_single_process_exact_batch_list。


def _ref_cached(d):
    """BucketBatchSampler._get_cached_dataset 的原样拷贝（该方法本次未改动）。"""
    if hasattr(d, "bucket_for_index"):
        return d
    if hasattr(d, "dataset"):
        return _ref_cached(d.dataset)
    return None


def _reference_iter(dataset, batch_size, drop_last, shuffle, seed, epoch):
    rng = random.Random(seed + epoch)
    cached = _ref_cached(dataset)
    out = []
    if cached is None:
        indices = list(range(len(dataset)))
        if shuffle:
            rng.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch = indices[i:i + batch_size]
            if len(batch) < batch_size and drop_last:
                continue
            out.append(batch)
        return out

    base_len = len(cached)
    bucket_to_indices = {}
    for idx in range(len(dataset)):
        bucket = cached.bucket_for_index[idx % base_len]
        if bucket is None:
            bucket = (0, 0)
        bucket_to_indices.setdefault(bucket, []).append(idx)
    buckets = list(bucket_to_indices.keys())
    if shuffle:
        rng.shuffle(buckets)
    for bucket in buckets:
        indices = bucket_to_indices[bucket]
        if shuffle:
            rng.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch = indices[i:i + batch_size]
            if len(batch) < batch_size and drop_last:
                continue
            out.append(batch)
    return out


def _reference_len(dataset, batch_size, drop_last):
    def _f(n):
        return n // batch_size if drop_last else (n + batch_size - 1) // batch_size

    cached = _ref_cached(dataset)
    if cached is None:
        return _f(len(dataset))
    counts = {}
    base_len = len(cached)
    for idx in range(len(dataset)):
        bucket = cached.bucket_for_index[idx % base_len]
        if bucket is None:
            bucket = (0, 0)
        counts[bucket] = counts.get(bucket, 0) + 1
    return sum(_f(n) for n in counts.values())


# ============================================== 1. 单进程逐字节不变（硬约束）

_MATRIX = [
    # (每桶样本数, batch_size, drop_last, shuffle, epoch)
    ([3, 5], 2, False, False, 0),
    ([3, 5], 2, True, False, 0),
    ([3, 5], 2, False, True, 0),
    ([3, 5], 2, False, True, 3),
    ([3, 5], 2, True, True, 7),
    ([30, 27, 25, 24, 23], 2, False, True, 0),
    ([30, 27, 25, 24, 23], 4, True, True, 1),
    ([1], 2, False, True, 0),
    ([1], 2, True, True, 0),
    ([9, 9, 9], 3, True, True, 2),
]


@pytest.mark.parametrize(("sizes", "bs", "drop_last", "shuffle", "epoch"), _MATRIX)
def test_single_process_matches_pre_ddp_reference(monkeypatch, sizes, bs, drop_last, shuffle, epoch):
    """world_size=1 时 __iter__ / __len__ 与改动前实现逐字节一致。

    这是整个改动的硬约束：不开多卡的用户（含全部 NVIDIA 单卡用户）走这条路。
    """
    ds = _MockBucketedDataset(_bucketed(sizes))

    def make():
        return BucketBatchSampler(ds, batch_size=bs, drop_last=drop_last,
                                  shuffle=shuffle, seed=42)

    with monkeypatch.context() as m:
        _fake_topology(m, 1, 0)
        sampler = make()
        sampler.set_epoch(epoch)
        assert [list(b) for b in sampler] == _reference_iter(ds, bs, drop_last, shuffle, 42, epoch)
        assert len(sampler) == _reference_len(ds, bs, drop_last)


def test_single_process_matches_reference_without_bucket_info(monkeypatch):
    """无桶信息的线性回退分支同样零回归。"""
    ds = _PlainDataset(10)
    for drop_last in (True, False):
        with monkeypatch.context() as m:
            _fake_topology(m, 1, 0)
            sampler = BucketBatchSampler(ds, batch_size=3, drop_last=drop_last,
                                         shuffle=True, seed=42)
            assert [list(b) for b in sampler] == _reference_iter(ds, 3, drop_last, True, 42, 0)
            assert len(sampler) == _reference_len(ds, 3, drop_last)


@pytest.mark.parametrize("ws", [1, 2, 4])
def test_repeat_wrapper_path(monkeypatch, ws):
    """RepeatDataset(CachedLatentDataset) 包装（Kohya 的 `5_concept` 目录）也要成立。

    这条路走 sampler 里的 ``idx % base_len``：外层索引比桶列表长，桶信息在内层。
    ws=1 比参照实现，ws>1 比分片不变式。
    """
    inner = _MockBucketedDataset(_bucketed([3, 5]))
    ds = _RepeatWrapper(inner, repeats=3)   # 24 个样本，8 个桶条目

    def make():
        return BucketBatchSampler(ds, batch_size=2, drop_last=False, shuffle=True, seed=42)

    if ws == 1:
        with monkeypatch.context() as m:
            _fake_topology(m, 1, 0)
            s = make()
            assert [list(b) for b in s] == _reference_iter(ds, 2, False, True, 42, 0)
            assert len(s) == _reference_len(ds, 2, False)
        return

    baseline = {tuple(b) for b in _single(make, monkeypatch)}
    shards = _shards(make, monkeypatch, ws)
    assert len({len(b) for b, _n in shards}) == 1
    for batches, n in shards:
        assert n == len(batches)
        assert {tuple(b) for b in batches} <= baseline
        for b in batches:
            _bucket_of(ds, b)


def test_single_process_exact_batch_list(monkeypatch):
    """硬编码一份具体输出，改语义时 diff 里一眼能看见（shuffle=False 才好读）。"""
    ds = _MockBucketedDataset(_bucketed([3, 5]))  # 桶(0,0)=idx0..2，桶(1,1)=idx3..7
    with monkeypatch.context() as m:
        _fake_topology(m, 1, 0)
        sampler = BucketBatchSampler(ds, batch_size=2, drop_last=False, shuffle=False)
        assert [list(b) for b in sampler] == [[0, 1], [2], [3, 4], [5, 6], [7]]
        assert len(sampler) == 5


def test_single_process_uses_real_distributed_module(monkeypatch):
    """不打 patch、只清掉 torchrun 环境变量：验证真 utils.distributed 接线正确。

    上面的测试都把 world_size/rank 换成了 lambda，接错 API（比如写成
    dist_env.get_world_size()）也照样过；这条走真函数。
    """
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    ds = _MockBucketedDataset(_bucketed([4, 6]))
    sampler = BucketBatchSampler(ds, batch_size=2, drop_last=False, shuffle=True, seed=42)
    assert [list(b) for b in sampler] == _reference_iter(ds, 2, False, True, 42, 0)
    assert len(sampler) == _reference_len(ds, 2, False) == 5


def test_shard_helper_is_identity_when_single_process(monkeypatch):
    """单进程下 _ddp_shard 连列表对象都不换 —— 不 copy、不重排、零开销。"""
    with monkeypatch.context() as m:
        _fake_topology(m, 1, 0)
        batches = [[0, 1], [2, 3], [4]]
        assert _ds._ddp_shard(batches, "test") is batches


def test_single_process_stays_on_the_lazy_path(monkeypatch):
    """单进程下 BucketBatchSampler 根本不进分片代码 —— 不物化完整 batch 列表。

    分片必须先知道总数、因此必须物化；单进程没有这个需要，几万张图的数据集上白占
    内存。这条钉住那个 `if not is_distributed(): yield from ...` 的短路分支：光看
    输出看不出差别（ws=1 时分片是恒等），所以直接断言 _ddp_shard 没被调用。
    （NavitPackBatchSampler 无此性质：打包本来就要物化整份包列表。）
    """
    ds = _MockBucketedDataset(_bucketed([4, 6]))

    def _boom(batches, label):
        raise AssertionError(f"单进程不该调用 _ddp_shard（label={label}）")

    with monkeypatch.context() as m:
        _fake_topology(m, 1, 0)
        m.setattr(_ds, "_ddp_shard", _boom)
        sampler = BucketBatchSampler(ds, batch_size=2, drop_last=False, shuffle=False)
        assert [list(b) for b in sampler] == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        assert len(sampler) == 5


# ======================================== 2. BucketBatchSampler 的分片不变式

_SHARD_CASES = [
    # (每桶样本数, batch_size, shuffle, world_size)
    ([8, 8], 2, False, 2),
    ([8, 8], 2, True, 2),
    ([8, 8, 8, 8], 2, True, 4),
    ([30, 27, 25, 24, 23], 2, True, 2),
    ([30, 27, 25, 24, 23], 2, True, 4),
    ([30, 27, 25, 24, 23], 4, True, 4),
    ([5], 2, True, 2),          # 单桶，总数 3 个 batch（不整除 2）
    ([7, 3], 3, True, 4),       # batch 数少、桶零头多
]


@pytest.mark.parametrize(("sizes", "bs", "shuffle", "ws"), _SHARD_CASES)
def test_all_ranks_get_equal_batch_count(monkeypatch, sizes, bs, shuffle, ws):
    """各 rank batch 数严格相等 —— 不等就是 DDP 死锁（少的先退出，其余等在 all_reduce）。"""
    ds = _MockBucketedDataset(_bucketed(sizes))

    def make():
        return BucketBatchSampler(ds, batch_size=bs, drop_last=False, shuffle=shuffle, seed=42)

    counts = {len(batches) for batches, _n in _shards(make, monkeypatch, ws)}
    assert len(counts) == 1, f"各 rank batch 数不等：{counts}"
    assert counts.pop() == len(_single(make, monkeypatch)) // ws


@pytest.mark.parametrize(("sizes", "bs", "shuffle", "ws"), _SHARD_CASES)
def test_len_matches_actual_iteration_per_rank(monkeypatch, sizes, bs, shuffle, ws):
    """__len__ 必须等于本 rank 真迭代出的数量（分片后），否则 loop.py 的
    梯度累积尾组判定会错 —— 尾组不 step，梯度泄漏到下个 epoch。"""
    ds = _MockBucketedDataset(_bucketed(sizes))

    def make():
        return BucketBatchSampler(ds, batch_size=bs, drop_last=False, shuffle=shuffle, seed=42)

    for batches, n in _shards(make, monkeypatch, ws):
        assert n == len(batches)


@pytest.mark.parametrize(("sizes", "bs", "shuffle", "ws"), _SHARD_CASES)
def test_shards_are_subset_of_single_process_batches(monkeypatch, sizes, bs, shuffle, ws):
    """所有 rank 的 batch 并集 ⊆ 单进程的 batch 集合 —— 分片不能凭空造 batch，
    也不能把样本重新组合进别的 batch（那会破坏分桶）。"""
    ds = _MockBucketedDataset(_bucketed(sizes))

    def make():
        return BucketBatchSampler(ds, batch_size=bs, drop_last=False, shuffle=shuffle, seed=42)

    baseline = {tuple(b) for b in _single(make, monkeypatch)}
    for batches, _n in _shards(make, monkeypatch, ws):
        assert {tuple(b) for b in batches} <= baseline


@pytest.mark.parametrize(("sizes", "bs", "shuffle", "ws"), _SHARD_CASES)
def test_no_batch_processed_by_two_ranks(monkeypatch, sizes, bs, shuffle, ws):
    """rank 间无重叠：同一个 batch 不会被两个 rank 各训一遍（会静默加权）。"""
    ds = _MockBucketedDataset(_bucketed(sizes))

    def make():
        return BucketBatchSampler(ds, batch_size=bs, drop_last=False, shuffle=shuffle, seed=42)

    seen = [tuple(b) for batches, _n in _shards(make, monkeypatch, ws) for b in batches]
    assert len(seen) == len(set(seen))
    # 样本级也不能重复：一张图一个 epoch 只训一次
    samples = [i for b in seen for i in b]
    assert len(samples) == len(set(samples))


@pytest.mark.parametrize(("sizes", "bs", "shuffle", "ws"), _SHARD_CASES)
def test_bucket_grouping_preserved_after_shard(monkeypatch, sizes, bs, shuffle, ws):
    """每个 batch 内所有样本同桶 —— 这是不能用 DistributedSampler 的根因：
    它按样本索引切，一个 batch 会混进不同分辨率，collate 的 torch.stack 直接炸。"""
    ds = _MockBucketedDataset(_bucketed(sizes))

    def make():
        return BucketBatchSampler(ds, batch_size=bs, drop_last=False, shuffle=shuffle, seed=42)

    for batches, _n in _shards(make, monkeypatch, ws):
        for b in batches:
            _bucket_of(ds, b)  # 内部断言同桶


@pytest.mark.parametrize("ws", [2, 3, 4, 5])
def test_truncates_to_total_floor_div_world_size(monkeypatch, ws):
    """总 batch 数不整除 world_size 时截断到 total // ws（丢尾，不补齐）。

    补齐（重复样本凑数）会让部分样本一个 epoch 训两次，静默改训练语义；
    截断最多丢 ws-1 个 batch。这里同时钉住"丢的量正好是余数"。
    """
    ds = _MockBucketedDataset(_bucketed([30, 27, 25, 24, 23]))  # 66 个 batch @ bs=2

    def make():
        return BucketBatchSampler(ds, batch_size=2, drop_last=False, shuffle=True, seed=42)

    total = len(_single(make, monkeypatch))
    assert total == 66
    shards = _shards(make, monkeypatch, ws)
    per_rank = total // ws
    assert all(len(b) == per_rank for b, _n in shards)
    kept = sum(len(b) for b, _n in shards)
    assert kept == per_rank * ws
    assert total - kept == total % ws <= ws - 1


def test_stride_slicing_spreads_buckets_not_contiguous_blocks(monkeypatch):
    """stride 切片让各 rank 的桶分布近似一致；连续块切法会让 rank 0 全拿小桶。

    batch 列表按桶顺序排（一桶的 batch 连续），连续块 ⇒ 显存/耗时严重不均，
    而 DDP 每步等最慢的 rank。这条同时把"不是连续块"写死。
    """
    ds = _MockBucketedDataset(_bucketed([4, 4, 4, 4]))  # 4 桶 × 2 batch = 8 个 batch

    def make():
        return BucketBatchSampler(ds, batch_size=2, drop_last=False, shuffle=False, seed=42)

    full = _single(make, monkeypatch)
    assert len(full) == 8
    shards = _shards(make, monkeypatch, 2)
    all_buckets = {(i, i) for i in range(4)}
    for r, (batches, _n) in enumerate(shards):
        assert {_bucket_of(ds, b) for b in batches} == all_buckets, "该 rank 的桶分布不全"
        contiguous = full[r * 4:(r + 1) * 4]
        assert batches != contiguous, "退化成连续块切片了"
    assert [list(b) for b in shards[0][0]] == full[0::2]
    assert [list(b) for b in shards[1][0]] == full[1::2]


def test_plain_dataset_without_buckets_is_also_sharded(monkeypatch):
    """无桶信息的线性回退分支也要分片（否则各 rank 重复训全量数据）。"""
    def make():
        return BucketBatchSampler(_PlainDataset(10), batch_size=3, drop_last=False,
                                  shuffle=False, seed=42)

    full = _single(make, monkeypatch)
    assert full == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    shards = _shards(make, monkeypatch, 2)
    assert [b for b, _n in shards] == [[[0, 1, 2], [6, 7, 8]], [[3, 4, 5], [9]]]
    assert [n for _b, n in shards] == [2, 2]


def test_set_epoch_reshuffles_and_all_ranks_rebuild_the_same_full_list(monkeypatch):
    """set_epoch 换顺序；同一 epoch 下各 rank 重建的**完整**列表一致。

    后半句是零通信分片的前提：种子只含 seed+epoch、不含 rank，所以各 rank 独立
    算出同一份完整列表，各自 stride 切自己那份就自动无重叠、无遗漏。验证手段是
    「并集 == 单进程列表截断到 per_rank*ws 后的集合」—— 若某 rank 的完整列表不同，
    它切出来的 batch 就会落在基线之外。
    """
    ds = _MockBucketedDataset(_bucketed([30, 27, 25, 24, 23]))

    def make():
        return BucketBatchSampler(ds, batch_size=2, drop_last=False, shuffle=True, seed=42)

    ws = 4
    for epoch in (0, 1, 5):
        full = _single(make, monkeypatch, epoch=epoch)
        per_rank = len(full) // ws
        shards = _shards(make, monkeypatch, ws, epoch=epoch)
        union = {tuple(b) for batches, _n in shards for b in batches}
        assert union == {tuple(b) for b in full[:per_rank * ws]}

    # 换 epoch → 顺序变（否则每 epoch 丢的都是同一批尾部样本）
    e0 = _single(make, monkeypatch, epoch=0)
    e1 = _single(make, monkeypatch, epoch=1)
    assert e0 != e1
    r0_e0 = _shards(make, monkeypatch, ws, epoch=0)[0][0]
    r0_e1 = _shards(make, monkeypatch, ws, epoch=1)[0][0]
    assert r0_e0 != r0_e1


def test_warns_when_batches_fewer_than_world_size(monkeypatch, caplog):
    """batch 数 < world_size → 截断后各 rank 都拿 0 个（空 epoch），必须 warn。"""
    ds = _MockBucketedDataset(_bucketed([3]))  # bs=2, drop_last=False → 2 个 batch

    def make():
        return BucketBatchSampler(ds, batch_size=2, drop_last=False, shuffle=False, seed=42)

    _ds._DDP_SHARD_LOGGED.clear()  # 日志按 (sampler, ws) 去重，清掉才不受测试顺序影响
    with caplog.at_level("WARNING"), monkeypatch.context() as m:
        _fake_topology(m, 8, 0)
        sampler = make()
        assert list(sampler) == []
        assert len(sampler) == 0
    assert "world_size=8" in caplog.text


# ====================================== 3. NavitPackBatchSampler 的分片不变式

_NAVIT_COUNTS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 5, 8, 12, 40, 6, 21]


def _navit_factory(counts=None, **kw):
    counts = list(counts if counts is not None else _NAVIT_COUNTS)

    def make():
        opts = {"token_budget": 50, "shuffle": True, "seed": 42}
        opts.update(kw)
        return NavitPackBatchSampler(_FakeTokenDataset(counts), **opts)

    return make


def test_navit_single_process_unchanged(monkeypatch):
    """单进程输出不变：shuffle=False 下硬编码，shuffle=True 下仍覆盖每个样本恰好一次。"""
    with monkeypatch.context() as m:
        _fake_topology(m, 1, 0)
        s = NavitPackBatchSampler(_FakeTokenDataset([30, 30, 30]), token_budget=60, shuffle=False)
        assert [list(p) for p in s] == [[0, 1], [2]]
        assert len(s) == 2

        s2 = _navit_factory()()
        packs = [list(p) for p in s2]
        assert sorted(i for p in packs for i in p) == list(range(len(_NAVIT_COUNTS)))
        assert len(s2) == len(packs)


@pytest.mark.parametrize("ws", [2, 3, 4])
@pytest.mark.parametrize("strategy", ["next_fit", "ffd"])
def test_navit_all_ranks_get_equal_pack_count(monkeypatch, ws, strategy):
    make = _navit_factory(strategy=strategy, ffd_window=4)
    counts = {len(p) for p, _n in _shards(make, monkeypatch, ws)}
    assert len(counts) == 1, f"各 rank 包数不等：{counts}"
    assert counts.pop() == len(_single(make, monkeypatch)) // ws


@pytest.mark.parametrize("ws", [2, 3, 4])
def test_navit_len_matches_iteration_per_rank(monkeypatch, ws):
    """__len__ 走 _cached_packs，必须与 __iter__ 同为分片后的数量。"""
    make = _navit_factory()
    for packs, n in _shards(make, monkeypatch, ws):
        assert n == len(packs)


@pytest.mark.parametrize("ws", [2, 3, 4])
def test_navit_shards_subset_and_disjoint(monkeypatch, ws):
    make = _navit_factory()
    baseline = {tuple(p) for p in _single(make, monkeypatch)}
    seen = [tuple(p) for packs, _n in _shards(make, monkeypatch, ws) for p in packs]
    assert set(seen) <= baseline
    assert len(seen) == len(set(seen))
    samples = [i for p in seen for i in p]
    assert len(samples) == len(set(samples))


@pytest.mark.parametrize("ws", [2, 3, 4])
def test_navit_len_then_iter_stay_consistent(monkeypatch, ws):
    """set_epoch 后先取 len 再迭代，两者必须一致（包数依赖打包顺序，两处各算一次
    很容易算出不同的数 → loop.py 的尾组判定错）。"""
    make = _navit_factory()
    for r in range(ws):
        with monkeypatch.context() as m:
            _fake_topology(m, ws, r)
            s = make()
            s.set_epoch(2)
            n_before = len(s)          # 先算 len（内部建包 + 分片并缓存）
            assert len([list(p) for p in s]) == n_before


def test_navit_truncation_amount(monkeypatch):
    """截断量 = 全局包数 % ws，且各 rank 等长。"""
    make = _navit_factory()
    total = len(_single(make, monkeypatch))
    for ws in (2, 3, 4, 5):
        shards = _shards(make, monkeypatch, ws)
        per_rank = total // ws
        assert all(len(p) == per_rank for p, _n in shards)
        assert total - per_rank * ws == total % ws


def test_navit_drop_last_applies_before_shard(monkeypatch):
    """顺序必须是「打包 → drop_last 去尾 → 分片」。

    counts=[60,60,60,5] @ budget=60 → 4 个包，尾包 [3]（sum=5<60）被 drop_last 去掉
    → 3 个包 → ws=2 截断到每 rank 1 个。若反过来（先分片再 drop_last）：rank0 拿
    [[0],[2]] 尾包满、不丢 → 2 个；rank1 拿 [[1],[3]] 尾包不满、丢掉 → 1 个 ⇒
    包数不等 ⇒ 死锁。这条就是钉这个顺序。
    """
    make = _navit_factory([60, 60, 60, 5], token_budget=60, shuffle=False, drop_last=True)
    assert _single(make, monkeypatch) == [[0], [1], [2]]
    shards = _shards(make, monkeypatch, 2)
    assert [p for p, _n in shards] == [[[0]], [[1]]]
    assert [n for _p, n in shards] == [1, 1]


def test_navit_set_epoch_all_ranks_rebuild_same_full_list(monkeypatch):
    """同一 epoch 下各 rank 重建的完整包列表一致（零通信分片的前提）；换 epoch 换顺序。"""
    make = _navit_factory()
    ws = 3
    for epoch in (0, 1, 4):
        full = _single(make, monkeypatch, epoch=epoch)
        per_rank = len(full) // ws
        union = {tuple(p) for packs, _n in _shards(make, monkeypatch, ws, epoch=epoch)
                 for p in packs}
        assert union == {tuple(p) for p in full[:per_rank * ws]}
    assert _single(make, monkeypatch, epoch=0) != _single(make, monkeypatch, epoch=1)
