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
import time
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
#
# 上游的 ``_ensure_nvml`` 在此不需要：本分支把采集委托给
# ``accelerator.device_stats()``（按后端选路径的单一权威源），NVML 的 init 与
# 失败缓存都在那一层内部完成。这里只保留「查不到就别再查」的闩锁语义，外加
# 上游新增的连续失败熔断（见 _FUSE_LIMIT）。
_probe_lock = threading.Lock()
_probe_state: dict[str, Any] = {"disabled": False, "ever_ok": False}


# ── active GPU（torch 实际在用的卡）解析 ───────────────────────────────
# 判定已下沉到 utils/accelerator.py（``_nvml_active_index`` + ``torch_device_pci_bus_id``）：
# 那里是「按后端选查询路径」的单一权威源，NVIDIA 走 NVML PCI 比对、DCU 走 torch 序号，
# 两个后端各自算。本层只透传 ``DeviceStats.active``。
#
# 上游在此处有一份 NVML 专用实现（_resolve_active_index / _torch_pci_bus_id /
# _env_selected_index）—— 本分支删掉它以免两份逻辑漂移；对应测试也已迁到
# tests/test_accelerator.py。

# ── 数据结构 ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GpuStats:
    """单卡快照。字段名与顺序即 API 的 JSON key 与顺序（``asdict`` 直出），
    前端 ``api/client.ts`` 在消费——**只加不改**。

    与 ``accelerator.DeviceStats`` 字段同名但**不复用**它：那是 utils 层的内部
    结构，字段顺序不同（JSON key 顺序会变），且 API schema 该由 studio 层自己
    拥有，免得 utils 的重构直接漏到前端契约上。

    除显存外都可能为 None，前端一律按可缺失渲染（``util_pct`` 靠解析 hy-smi 文本、
    功率与频率靠 sysfs，都是 best-effort，详 ``accelerator._torch_device_stats``）。
    """

    index: int
    name: str
    #: 加速器利用率百分比；解析不到时 None（前端需按可缺失渲染）
    util_pct: Optional[int]
    vram_used_gb: float
    vram_total_gb: float
    temp_c: Optional[int] = None
    #: torch 实际在用的卡（多卡机器前端显示这张，而不是盲选 gpu[0]）。
    #: 解析不出（CPU-only torch / PCI 匹配失败）时全 False，前端回退 gpu[0]。
    active: bool = False
    #: 实时功率（瓦）。**只有 DCU 有**，走 sysfs `power1_average`——
    #: hy-smi 的 AvgPwr 列实测偏差 6-7 倍，不能用。
    power_w: Optional[int] = None
    #: 功率上限（瓦），sysfs `power1_cap_max`。是天花板不是目标：真实负载点亮的是
    #: 芯片不同部分，平均功率远低于上限属正常。
    power_cap_w: Optional[int] = None
    #: 当前核心频率（MHz）。要配 `sclk_max_mhz` 看才有意义。
    sclk_mhz: Optional[int] = None
    #: 最高档核心频率（MHz）。`sclk_mhz == sclk_max_mhz` = 没降频，是健康状态；
    #: 撞功率墙或温度墙的卡会主动降档 —— 这比看功率数字可靠。
    sclk_max_mhz: Optional[int] = None


@dataclass(frozen=True)
class SystemStats:
    cpu_pct: float
    ram_used_gb: float
    ram_total_gb: float
    # None = 查不到加速器（无卡 / 驱动缺失 / torch 不可用）；[] = 后端可用但 0 卡
    # (前端两种都隐藏 GPU pill)
    gpu: Optional[list[GpuStats]]


# ── 采集 ─────────────────────────────────────────────────────────────
# 采集失败熔断状态（R8）：list 单元素当可变 cell 用（函数内免 global 声明）。
_FUSE_LIMIT = 3
_GPU_FAILS = [0]
_GPU_DISABLED = [False]
_PSUTIL_FAILS = [0]
_PSUTIL_DISABLED = [False]


def _bytes_to_gb(n: int) -> float:
    return round(n / (1024 ** 3), 2)


def _collect_gpu() -> Optional[list[GpuStats]]:
    """逐卡快照；查不到返回 None（前端隐藏 GPU pill），0 卡返回 []。

    两道保护叠加：
    - **闩锁**（``_probe_state``）：从未成功过就永久关闭 —— CPU 机器不该每 2.5s
      试一次并刷日志。
    - **熔断**（``_GPU_FAILS`` / ``_FUSE_LIMIT``，上游 R8）：曾经成功过但连续失败
      3 次也停采。采样 2.5s 一 tick，坏死时逐条 exception 是 ~1440 条带
      traceback/小时，能吃穿 studio.log 配额。

    ``device_stats()`` 内部已把所有失败吃成 None，所以这里只需处理三态映射；
    外层 try/except 只兜底真正意外的异常（accelerator 自身抛了没预料到的东西）。
    """
    if _probe_state["disabled"] or _GPU_DISABLED[0]:
        return None
    try:
        # 上游这里原本自己跑 NVML 循环（含 active 判定）。改为委托 accelerator：
        # NVML 是 NVIDIA 专有，海光 DCU 上 nvmlDeviceGetCount 直接抛，于是整个
        # GPU 监控在 DCU 上永久关闭。accelerator 是「按后端选查询路径」的单一权威源
        # （NVIDIA→NVML、DCU→torch mem_get_info + hy-smi + sysfs），active 判定也
        # 一并下沉到那里（两个后端各自算，见 DeviceStats.active）。
        stats = accelerator.device_stats()
    except Exception:  # noqa: BLE001  采集不该让轮询接口 500
        # 熔断计数（上游 R8）：首次 WARNING 全文，连续 _FUSE_LIMIT 次后停采。
        _GPU_FAILS[0] += 1
        if _GPU_FAILS[0] == 1:
            logger.warning("gpu stats collection failed", exc_info=True)
        elif _GPU_FAILS[0] >= _FUSE_LIMIT and not _GPU_DISABLED[0]:
            _GPU_DISABLED[0] = True
            logger.warning(
                "gpu stats collection failed %d times in a row, sampling disabled "
                "for this process (restart to recover)", _GPU_FAILS[0],
            )
        stats = None
    else:
        _GPU_FAILS[0] = 0

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
            active=d.active,
            power_w=d.power_w,
            power_cap_w=d.power_cap_w,
            sclk_mhz=d.sclk_mhz,
            sclk_max_mhz=d.sclk_max_mhz,
        )
        for d in stats
    ]


def collect_stats() -> SystemStats:
    if _PSUTIL_DISABLED[0]:
        return SystemStats(cpu_pct=0.0, ram_used_gb=0.0, ram_total_gb=0.0, gpu=_collect_gpu())
    try:
        # interval=None: 返回自上次调用以来的 CPU 占用；首次调用返回 0.0，
        # 后续轮询拿到的就是 2-3s 平均值，对实时监控刚好。
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_used = _bytes_to_gb(mem.total - mem.available)
        ram_total = _bytes_to_gb(mem.total)
        _PSUTIL_FAILS[0] = 0
    except Exception:
        # 熔断（R8）：同 _collect_gpu——psutil 挂掉是环境级问题，重试无意义
        _PSUTIL_FAILS[0] += 1
        if _PSUTIL_FAILS[0] == 1:
            logger.warning("psutil stats collection failed", exc_info=True)
        elif _PSUTIL_FAILS[0] >= _FUSE_LIMIT and not _PSUTIL_DISABLED[0]:
            _PSUTIL_DISABLED[0] = True
            logger.warning(
                "psutil stats collection failed %d times in a row, reporting "
                "zeros for this process (restart to recover)", _PSUTIL_FAILS[0],
            )
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
        # tick 兜底节流（R8）：首条 ERROR 全文（这层通常是 bus/序列化 bug），
        # 之后每 60s 一条计数汇总；不熔断——下游可能恢复。
        fail_count = 0
        last_report = 0.0
        while not self._stop.is_set():
            try:
                payload = stats_to_json(collect_stats())
                self._on_sample(payload)
                if fail_count:
                    logger.info("system stats sampler recovered after %d failed tick(s)", fail_count)
                    fail_count = 0
            except Exception:
                fail_count += 1
                now = time.monotonic()
                if fail_count == 1:
                    logger.exception("system stats sampler tick failed")
                    last_report = now
                elif now - last_report >= 60.0:
                    logger.warning("system stats sampler still failing: %d tick(s) since last report", fail_count)
                    last_report = now
            self._stop.wait(self._interval)
