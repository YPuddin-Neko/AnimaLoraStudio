"""系统资源采集 (CPU / RAM / GPU / VRAM)。

供 topbar 实时小组件按 2-3s 轮询使用。

设计：
    - GPU 段委托 ``utils.accelerator.device_stats()``（单一权威源）：NVIDIA 走
      NVML（利用率 / 温度齐全），海光 DCU 走 torch ``mem_get_info``（NVML 是
      NVIDIA 专有，DCU 上根本 init 不了，只有显存拿得到）。本模块**不**自己判
      后端、不自己碰 pynvml。
    - 首次探测失败 (无加速器 / 驱动缺失 / 库未装) **永久标记**，之后直接返回
      gpu=None — 不重试、不刷日志。这是 2-3s 轮询的热路径，而
      ``device_stats()`` 自身无缓存、每次调用都会重试 NVML init，CPU 机器上放
      任它跑就是每 2.5s 一次无谓的 init + 一行日志。
    - psutil 几乎不会失败；仍 try/except 兜底，让前端轮询不会因偶发问题挂掉。
    - 模块无状态导出，调用 collect_stats() 即可。
"""
from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import psutil

from utils import accelerator

logger = logging.getLogger(__name__)

# psutil.cpu_percent(interval=None) 第一次调用返回 0.0 (无 baseline)，
# 之后返回「距上次调用以来」的平均占用。模块导入时 prime 一下，让首请求
# 就能拿到从启动到首请求的平均值，避免前端首次轮询永远显示 0%。
psutil.cpu_percent(interval=None)


# ── GPU 探测闩锁 ──────────────────────────────────────────────────────
#: ``disabled`` 一旦置 True 就不再调 ``device_stats()``；``ever_ok`` 记录是否曾
#: 成功过。两个字段而不是一个的原因：要区分「这台机器没有加速器」与「有卡但这
#: 一拍查询抖了一下」。前者第一次就失败 → 永久关掉（CPU 机器不该每 2.5s 试一
#: 次 NVML init 并刷日志，原 pynvml 实现刻意做了这件事，语义在此保留）；后者曾
#: 经成功过 → 只是本拍返回 None，下一拍照常重试，不能因为一次抖动就让 GPU pill
#: 永久消失。
_probe_lock = threading.Lock()
_probe_state: dict[str, Any] = {"disabled": False, "ever_ok": False}


# ── 数据结构 ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GpuStats:
    """单卡快照。字段名与顺序即 API 的 JSON key 与顺序（``asdict`` 直出），
    前端 ``api/client.ts`` 在消费——**只加不改**。

    与 ``accelerator.DeviceStats`` 字段同名但**不复用**它：那是 utils 层的内部
    结构，字段顺序不同（JSON key 顺序会变），且 API schema 该由 studio 层自己
    拥有，免得 utils 的重构直接漏到前端契约上。

    ``util_pct`` 可能为 None：DCU 上利用率要靠解析 hy-smi 文本，没有稳定的机器
    可读接口，accelerator 侧当前不报（详 ``accelerator._torch_device_stats``）。
    """

    index: int
    name: str
    #: 加速器利用率百分比；DCU 上暂为 None（前端需按可缺失渲染）
    util_pct: Optional[int]
    vram_used_gb: float
    vram_total_gb: float
    temp_c: Optional[int] = None


@dataclass(frozen=True)
class SystemStats:
    cpu_pct: float
    ram_used_gb: float
    ram_total_gb: float
    # None = 查不到加速器（无卡 / 驱动缺失 / torch 不可用）；[] = 后端可用但 0 卡
    # (前端两种都隐藏 GPU pill)
    gpu: Optional[list[GpuStats]]


# ── 采集 ─────────────────────────────────────────────────────────────
def _bytes_to_gb(n: int) -> float:
    return round(n / (1024 ** 3), 2)


def _collect_gpu() -> Optional[list[GpuStats]]:
    """逐卡快照；查不到返回 None（前端隐藏 GPU pill），0 卡返回 []。

    闩锁语义见 ``_probe_state``。``device_stats()`` 内部已把所有失败吃成 None，
    所以这里只需处理三态映射；外层 try/except 只兜底真正意外的异常（例如
    accelerator 自身抛了没预料到的东西），失败同样走闩锁判定。
    """
    if _probe_state["disabled"]:
        return None
    try:
        stats = accelerator.device_stats()
    except Exception:  # noqa: BLE001  采集不该让轮询接口 500
        logger.exception("gpu stats collection failed")
        stats = None

    if stats is None:
        # 只有「从未成功过」才永久关闭；日志一行，且靠锁保证只打一次。
        with _probe_lock:
            if not _probe_state["ever_ok"] and not _probe_state["disabled"]:
                _probe_state["disabled"] = True
                logger.info(
                    "加速器指标不可用，GPU 监控已关闭（后端: %s）",
                    accelerator.detect().vendor_label,
                )
        return None

    _probe_state["ever_ok"] = True
    return [
        GpuStats(
            index=d.index,
            name=d.name,
            util_pct=d.util_pct,
            vram_used_gb=d.vram_used_gb,
            vram_total_gb=d.vram_total_gb,
            temp_c=d.temp_c,
        )
        for d in stats
    ]


def collect_stats() -> SystemStats:
    try:
        # interval=None: 返回自上次调用以来的 CPU 占用；首次调用返回 0.0，
        # 后续轮询拿到的就是 2-3s 平均值，对实时监控刚好。
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_used = _bytes_to_gb(mem.total - mem.available)
        ram_total = _bytes_to_gb(mem.total)
    except Exception:
        logger.exception("psutil stats collection failed")
        cpu = 0.0
        ram_used = 0.0
        ram_total = 0.0
    return SystemStats(
        cpu_pct=round(float(cpu), 1),
        ram_used_gb=ram_used,
        ram_total_gb=ram_total,
        gpu=_collect_gpu(),
    )


def stats_to_json(s: SystemStats) -> dict[str, Any]:
    return {
        "cpu_pct": s.cpu_pct,
        "ram_used_gb": s.ram_used_gb,
        "ram_total_gb": s.ram_total_gb,
        "gpu": [asdict(g) for g in s.gpu] if s.gpu is not None else None,
    }


# ── SSE sampler ──────────────────────────────────────────────────────
class SystemStatsSampler:
    """后台线程：周期性采集系统资源 → callback (通常是 bus.publish)。

    取代每个客户端独立轮询 /api/system/stats — 云部署场景下避免污染
    server access log、DevTools Network 面板、跨公网 RTT 开销。前端只在
    mount 时 GET 一次冷启动，之后走 SSE 持续接收。
    """

    def __init__(
        self,
        on_sample: Callable[[dict[str, Any]], None],
        *,
        interval: float = 2.5,
    ) -> None:
        self._on_sample = on_sample
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(
            target=self._run, name="system-stats-sampler", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                payload = stats_to_json(collect_stats())
                self._on_sample(payload)
            except Exception:
                logger.exception("system stats sampler tick failed")
            self._stop.wait(self._interval)
