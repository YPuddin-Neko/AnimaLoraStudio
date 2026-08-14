"""services/system_stats.py — 采集 + 加速器探测优雅降级 + SSE sampler 线程。

ADR 0016 起 GPU 采集委托 `utils.accelerator.device_stats()`（按后端选 NVML / torch），
所以本文件不再 mock pynvml —— 那层的字段映射由 `tests/test_accelerator.py` 覆盖。
这里只测本模块自己的职责：三态映射（None / [] / 非空）与「探测失败不重试」闩锁。
"""
from __future__ import annotations

import threading
import time

import pytest

from studio.services import system_stats
from utils.accelerator import DeviceStats


@pytest.fixture(autouse=True)
def _reset_probe_latch():
    """清 `_probe_state` 闩锁。

    该闩锁是模块级的、刻意跨调用保持（首次探测失败后永久关闭 GPU 采集，避免
    2-3s 轮询热路径上反复重试）。测试间不清会互相污染：任一用例触发永久关闭后，
    后续用例的 `_collect_gpu()` 会直接返回 None。
    """
    system_stats._probe_state.update({"disabled": False, "ever_ok": False})
    yield
    system_stats._probe_state.update({"disabled": False, "ever_ok": False})


def test_collect_stats_returns_sane_basic():
    """实环境采集：CPU/RAM 范围合理，结构完整。"""
    stats = system_stats.collect_stats()
    assert 0.0 <= stats.cpu_pct <= 100.0
    assert stats.ram_used_gb >= 0.0
    assert stats.ram_total_gb > 0.0
    assert stats.ram_used_gb <= stats.ram_total_gb
    # gpu 字段在 CI 环境通常是 None；本地有卡时是 list[GpuStats]


def test_stats_to_json_no_gpu():
    s = system_stats.SystemStats(
        cpu_pct=12.5, ram_used_gb=8.0, ram_total_gb=32.0, gpu=None,
    )
    j = system_stats.stats_to_json(s)
    assert set(j.keys()) == {"cpu_pct", "ram_used_gb", "ram_total_gb", "gpu"}
    assert j["gpu"] is None


def test_stats_to_json_with_gpu():
    g = system_stats.GpuStats(
        index=0, name="Test GPU", util_pct=42,
        vram_used_gb=4.0, vram_total_gb=24.0, temp_c=55,
    )
    s = system_stats.SystemStats(
        cpu_pct=1.0, ram_used_gb=8.0, ram_total_gb=32.0, gpu=[g],
    )
    j = system_stats.stats_to_json(s)
    assert isinstance(j["gpu"], list) and len(j["gpu"]) == 1
    g0 = j["gpu"][0]
    assert g0["index"] == 0
    assert g0["name"] == "Test GPU"
    assert g0["util_pct"] == 42
    assert g0["temp_c"] == 55


def test_probe_failure_returns_none_and_latches(monkeypatch: pytest.MonkeyPatch):
    """探测不可用（device_stats 返回 None）→ 永久关闭，不再调下去。

    「不再调」是本闩锁的**目的**而非细节：这是 2-3s 轮询的热路径，而
    `device_stats()` 自身无缓存、每次都会重试 NVML init。用调用计数断言，
    退化成每拍重试时这个用例会失败。
    """
    calls = [0]

    def never_available():
        calls[0] += 1
        return None

    monkeypatch.setattr(system_stats.accelerator, "device_stats", never_available)

    assert system_stats._collect_gpu() is None
    assert system_stats._collect_gpu() is None
    assert calls[0] == 1, "闩锁失效：失败后仍在重试探测"


def test_transient_failure_after_success_does_not_latch(monkeypatch: pytest.MonkeyPatch):
    """曾成功过之后偶发返回 None → 只是本拍没数据，下一拍照常重试。

    与上一个用例是一对：区分「这台机器没有加速器」（第一次就失败 → 永久关）与
    「有卡但这一拍查询抖了一下」（曾成功 → 不能因一次抖动让 GPU pill 永久消失）。
    """
    seq = [
        [DeviceStats(index=0, name="Card", vram_used_gb=1.0, vram_total_gb=8.0)],
        None,
        [DeviceStats(index=0, name="Card", vram_used_gb=2.0, vram_total_gb=8.0)],
    ]
    monkeypatch.setattr(
        system_stats.accelerator, "device_stats", lambda: seq.pop(0),
    )

    first = system_stats._collect_gpu()
    assert first is not None and first[0].vram_used_gb == 1.0
    assert system_stats._collect_gpu() is None          # 抖动这一拍
    third = system_stats._collect_gpu()                  # 不该被闩锁挡住
    assert third is not None and third[0].vram_used_gb == 2.0


def test_zero_devices_returns_empty_list(monkeypatch: pytest.MonkeyPatch):
    """后端可用但没卡：返回 [] (前端跟 None 一样隐藏 GPU pill)。"""
    monkeypatch.setattr(system_stats.accelerator, "device_stats", lambda: [])
    assert system_stats._collect_gpu() == []


def test_device_stats_fields_map_through(monkeypatch: pytest.MonkeyPatch):
    """accelerator.DeviceStats → GpuStats 逐字段映射（含可缺失的两项）。"""
    monkeypatch.setattr(
        system_stats.accelerator, "device_stats",
        lambda: [DeviceStats(
            index=0, name="Mock GPU", vram_used_gb=4.0, vram_total_gb=24.0,
            util_pct=67, temp_c=50, active=True,
        )],
    )
    result = system_stats._collect_gpu()
    assert result is not None and len(result) == 1
    g = result[0]
    assert (g.index, g.name, g.util_pct, g.temp_c) == (0, "Mock GPU", 67, 50)
    assert (g.vram_used_gb, g.vram_total_gb) == (4.0, 24.0)
    # active 由 accelerator 判定后透传（单卡短路恒真）。本文件的 fixture 直接给
    # DeviceStats，所以这里断言的是「透传不丢字段」；判定逻辑本身在
    # tests/test_accelerator.py 测。
    assert g.active is True


def test_missing_util_and_temp_pass_through_as_none(monkeypatch: pytest.MonkeyPatch):
    """DCU 口径：利用率 / 温度拿不到时保持 None，显存照常出。

    前端据此隐藏利用率 pill（`SystemStats.tsx`）——不能用 0 兜底，0% 是合法读数。
    """
    monkeypatch.setattr(
        system_stats.accelerator, "device_stats",
        lambda: [DeviceStats(
            index=0, name="Hygon BW1000", vram_used_gb=3.5, vram_total_gb=64.0,
        )],
    )
    result = system_stats._collect_gpu()
    assert result is not None
    assert result[0].util_pct is None
    assert result[0].temp_c is None
    assert result[0].vram_total_gb == 64.0
    # JSON 里必须是 null 而不是被丢掉 —— 前端按 `!= null` 判
    payload = system_stats.stats_to_json(system_stats.SystemStats(
        cpu_pct=1.0, ram_used_gb=8.0, ram_total_gb=32.0, gpu=result,
    ))
    assert payload["gpu"][0]["util_pct"] is None


def test_sampler_emits_payloads(monkeypatch: pytest.MonkeyPatch):
    """SystemStatsSampler 启动后会定期 callback；stop() 干净退出。"""
    samples: list[dict] = []
    event = threading.Event()

    def on_sample(payload: dict) -> None:
        samples.append(payload)
        if len(samples) >= 2:
            event.set()

    sampler = system_stats.SystemStatsSampler(on_sample, interval=0.05)
    sampler.start()
    try:
        assert event.wait(timeout=2.0), f"only got {len(samples)} samples"
    finally:
        sampler.stop()

    assert len(samples) >= 2
    for p in samples:
        assert set(p.keys()) == {"cpu_pct", "ram_used_gb", "ram_total_gb", "gpu"}


def test_sampler_swallows_collection_errors(monkeypatch: pytest.MonkeyPatch):
    """采集抛错时 sampler 不应崩溃，继续下一轮。"""
    fail_count = [0]
    samples: list[dict] = []

    real_collect = system_stats.collect_stats

    def flaky_collect() -> system_stats.SystemStats:
        fail_count[0] += 1
        if fail_count[0] == 1:
            raise RuntimeError("simulated transient failure")
        return real_collect()

    monkeypatch.setattr(system_stats, "collect_stats", flaky_collect)

    sampler = system_stats.SystemStatsSampler(samples.append, interval=0.05)
    sampler.start()
    try:
        deadline = time.time() + 2.0
        while len(samples) < 1 and time.time() < deadline:
            time.sleep(0.05)
    finally:
        sampler.stop()

    # 第一次 collect 抛错被吞，第二次成功 → samples >= 1
    assert fail_count[0] >= 2
    assert len(samples) >= 1


def test_collect_gpu_swallows_unexpected_exception(monkeypatch: pytest.MonkeyPatch):
    """device_stats 抛了没预料到的异常时也不能让轮询接口 500。

    `device_stats()` 契约上把失败吃成 None，本用例守的是「契约被破坏时仍然安全」
    —— /api/system/stats 是 2-3s 轮询端点，抛异常会在前端刷出连续报错。
    """
    def boom():
        raise RuntimeError("simulated accelerator blowup")

    monkeypatch.setattr(system_stats.accelerator, "device_stats", boom)
    assert system_stats._collect_gpu() is None
