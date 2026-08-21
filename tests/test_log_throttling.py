"""刀 1 机制修复：_ErrorThrottle / event_bus 丢弃状态化 / system_stats 熔断。"""
from __future__ import annotations

import asyncio
import logging
import sys
from types import SimpleNamespace

import pytest


# ── _ErrorThrottle ───────────────────────────────────────────────────
class TestErrorThrottle:
    def _make(self, monkeypatch, now: list[float]):
        from studio.supervisor.core import _ErrorThrottle
        import studio.supervisor.core as core
        monkeypatch.setattr(core.time, "monotonic", lambda: now[0])
        return _ErrorThrottle("test site")

    def test_first_full_then_silent_within_window(self, monkeypatch, caplog):
        now = [1000.0]
        th = self._make(monkeypatch, now)
        with caplog.at_level(logging.DEBUG, logger="studio.supervisor.core"):
            for _ in range(5):
                try:
                    raise ValueError("x")
                except ValueError:
                    th.failed()
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.ERROR  # exception

    def test_window_summary_and_recovery(self, monkeypatch, caplog):
        now = [1000.0]
        th = self._make(monkeypatch, now)
        with caplog.at_level(logging.DEBUG, logger="studio.supervisor.core"):
            for _ in range(3):
                try:
                    raise ValueError("x")
                except ValueError:
                    th.failed()
            now[0] += 61.0
            try:
                raise ValueError("x")
            except ValueError:
                th.failed()
            th.recovered()
        levels = [r.levelno for r in caplog.records]
        assert levels == [logging.ERROR, logging.WARNING, logging.INFO]
        assert "count=4" in caplog.records[1].message

    def test_exception_type_change_relogs_full(self, monkeypatch, caplog):
        now = [1000.0]
        th = self._make(monkeypatch, now)
        with caplog.at_level(logging.DEBUG, logger="studio.supervisor.core"):
            try:
                raise ValueError("x")
            except ValueError:
                th.failed()
            try:
                raise KeyError("y")
            except KeyError:
                th.failed()
        assert [r.levelno for r in caplog.records] == [logging.ERROR, logging.ERROR]


# ── event_bus 丢弃状态化 ─────────────────────────────────────────────
class TestEventBusDropState:
    def test_one_congestion_two_lines(self, caplog):
        from studio.infrastructure.event_bus import _safe_put

        async def scenario():
            q: asyncio.Queue = asyncio.Queue(maxsize=1)
            with caplog.at_level(logging.WARNING, logger="studio.infrastructure.event_bus"):
                _safe_put(q, {"type": "a"})          # 放进去
                for i in range(10):                   # 连续丢 10 个
                    _safe_put(q, {"type": f"drop{i}"})
                q.get_nowait()                        # 消费者恢复
                _safe_put(q, {"type": "b"})          # 恢复 → 汇总条
        asyncio.run(scenario())
        msgs = [r.message for r in caplog.records]
        assert len(msgs) == 2
        assert "dropping events" in msgs[0]
        assert "dropped 10 event(s)" in msgs[1]
        assert "drop9" in msgs[1]  # last_type


# ── system_stats 熔断 ────────────────────────────────────────────────
@pytest.fixture()
def reset_stats_state():
    """把 system_stats 的进程级熔断 / 闩锁状态复位（前后各一次）。

    `_probe_state` 也要复位：它是本分支特有的「从未成功过就永久关闭」闩锁
    （见 system_stats._collect_gpu），测试里会改 `ever_ok`，不复位会漏给后续用例。
    """
    import studio.services.system_stats as ss

    def _reset():
        ss._GPU_FAILS[0] = 0
        ss._GPU_DISABLED[0] = False
        ss._PSUTIL_FAILS[0] = 0
        ss._PSUTIL_DISABLED[0] = False
        ss._probe_state["disabled"] = False
        ss._probe_state["ever_ok"] = False

    _reset()
    yield ss
    _reset()


class TestSystemStatsFuse:
    def test_gpu_fuse_after_three_failures(self, reset_stats_state, monkeypatch, caplog):
        """采集连续抛异常 → 首条 WARNING + 熔断条，之后零日志。

        本分支的 `_collect_gpu` 委托 `accelerator.device_stats()`（按后端选路径的
        单一权威源），不再自己跑 NVML —— 上游那版 stub `_ensure_nvml` + 造残缺
        pynvml 的手法在这里打不中任何东西。改为让 device_stats 直接抛：熔断计数
        在 `_collect_gpu` 的 except 里，与数据源无关。

        `ever_ok=True` 是必需的前置：不设的话第一次失败就被「从未成功过」的闩锁
        永久关掉（返回 None 但 _GPU_DISABLED 仍是 False），测不到熔断那条路径。
        """
        ss = reset_stats_state
        ss._probe_state["ever_ok"] = True

        def boom():
            raise RuntimeError("accelerator down")

        monkeypatch.setattr(ss.accelerator, "device_stats", boom)
        with caplog.at_level(logging.DEBUG, logger="studio.services.system_stats"):
            for _ in range(5):
                assert ss._collect_gpu() is None
        assert ss._GPU_DISABLED[0] is True
        msgs = [r.message for r in caplog.records]
        assert len(msgs) == 2  # 首条 + 熔断条，之后零日志
        assert "sampling disabled" in msgs[1]

    def test_psutil_fuse_reports_zeros(self, reset_stats_state, monkeypatch, caplog):
        ss = reset_stats_state
        def boom(*a, **kw):
            raise RuntimeError("psutil down")
        monkeypatch.setattr(ss.psutil, "cpu_percent", boom)
        monkeypatch.setattr(ss, "_collect_gpu", lambda: None)
        with caplog.at_level(logging.DEBUG, logger="studio.services.system_stats"):
            for _ in range(5):
                s = ss.collect_stats()
        assert s.cpu_pct == 0.0
        assert ss._PSUTIL_DISABLED[0] is True
        msgs = [r.message for r in caplog.records]
        assert len(msgs) == 2
        assert "reporting zeros" in msgs[1]
