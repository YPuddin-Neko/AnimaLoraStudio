"""加速器后端识别（NVIDIA CUDA / 海光 DCU / CPU）—— 全仓库单一权威源。

**为什么需要这一层**：PyTorch 在 ROCm 系（海光 DTK 是 ROCm 分支）上把 CUDA API
映射到 HIP，所以 ``torch.cuda.*``、``torch.device("cuda")``、Stream / Event /
pinned memory 这些**照常可用**——训练主链路不需要改。真正会误判的是**装包与
检测层**，因为两者的版本标签长得不一样：

===============  ======================  ==========================
                 NVIDIA wheel            海光 DTK wheel
===============  ======================  ==========================
version.cuda     ``"12.8"``              ``None``
version.hip      ``None``                ``"6.3.42134-..."``
smi 工具         ``nvidia-smi``          ``hy-smi``
wheel 来源       download.pytorch.org    镜像内预装（不可 pip 覆盖）
===============  ======================  ==========================

只看 ``torch.version.cuda is None`` 就判「CPU-only 误装」会把 DCU 归错类：
启动期弹「检测到 GPU 但装了 CPU 版 torch」大警告、flash_attn 安装被拒、
Settings 里推荐重装 cu128 wheel——最后一项在 DTK 镜像上是**破坏性**的
（pip 覆盖掉镜像预装的 DTK torch，环境直接报废）。

**用法**：需要判断后端的地方一律问本模块，不要自己读 ``torch.version.*``。

    from utils.accelerator import detect, is_dcu

    info = detect()
    if info.backend == "dcu":
        ...

检测结果**进程内缓存**（``detect()`` 幂等）：torch build 在进程生命周期里不会变。

依赖方向（``docs/AGENTS.md`` §3.1）：本模块位于 ``utils/``，是依赖链最底层，
``runtime/`` 与 ``studio/`` 都可 import；本模块**不** import 它们。torch 是可选
依赖——bootstrap 阶段（venv 里只有 pip）也要能调 ``probe_stdlib()``。
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

#: 后端标识。``cuda`` = NVIDIA、``dcu`` = 海光 DCU（DTK / HIP）、``cpu`` = 无加速器。
#: 刻意不叫 ``rocm``：本项目只对海光 DCU 做过验证，AMD Radeon / Instinct 虽然同为
#: HIP build 但驱动栈与 smi 工具不同，没验证过的东西不在这里承诺支持。
Backend = Literal["cuda", "dcu", "cpu"]

#: 各后端的 smi 工具候选，按优先级。海光 DTK 装 ``hy-smi``；因为 DTK 是 ROCm
#: 分支，机器上**可能**同时有 ``rocm-smi``，作为次选。
_SMI_CANDIDATES: dict[Backend, tuple[str, ...]] = {
    "cuda": ("nvidia-smi",),
    "dcu": ("hy-smi", "rocm-smi"),
    "cpu": (),
}

#: 面向用户的后端名（日志 / UI / 报错文案统一走这里，避免各处自己拼字符串）。
VENDOR_LABEL: dict[Backend, str] = {
    "cuda": "NVIDIA CUDA",
    "dcu": "海光 DCU (DTK)",
    "cpu": "CPU",
}


@dataclass(frozen=True)
class AcceleratorInfo:
    """当前进程看到的加速器事实。字段全部只读，由 :func:`detect` 填充。"""

    backend: Backend
    #: ``torch.__version__``；torch 未装时 None
    torch_version: Optional[str] = None
    #: ``torch.version.cuda``（NVIDIA wheel 才有）
    cuda_version: Optional[str] = None
    #: ``torch.version.hip``（DTK / ROCm wheel 才有）
    hip_version: Optional[str] = None
    #: ``torch.cuda.is_available()``——装了对的 wheel 也可能因驱动 / 设备节点没挂而 False
    available: bool = False
    device_count: int = 0
    #: 逐卡名称，如 ``["Hygon BW1000"]`` / ``["NVIDIA GeForce RTX 4090"]``
    device_names: tuple[str, ...] = ()
    #: HIP 侧的 ``gcnArchName``（如 ``gfx928``）；NVIDIA 上为空
    gcn_arch: tuple[str, ...] = ()
    #: torch 装了但 import 失败时的原因（DLL / so 缺失等），供 UI 显示
    import_error: Optional[str] = None

    @property
    def is_gpu(self) -> bool:
        """有可用的加速器（不区分厂商）。"""
        return self.backend != "cpu" and self.available

    @property
    def vendor_label(self) -> str:
        return VENDOR_LABEL[self.backend]

    @property
    def torch_device(self) -> str:
        """训练 / 推理该用的 torch 设备字符串。

        DCU 上**同样是** ``"cuda"``——HIP build 把 CUDA API 名字整套复用，
        写 ``"hip"`` 反而会 ``RuntimeError``。这个 property 存在的意义是让调用方
        不必自己判断，而不是真的会返回第三个值。
        """
        return "cuda" if self.is_gpu else "cpu"

    def as_dict(self) -> dict[str, Any]:
        """给 API / 日志用的 JSON 友好形式。"""
        return {
            "backend": self.backend,
            "vendor_label": self.vendor_label,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "hip_version": self.hip_version,
            "available": self.available,
            "device_count": self.device_count,
            "device_names": list(self.device_names),
            "gcn_arch": list(self.gcn_arch),
            "import_error": self.import_error,
        }


_CACHE: Optional[AcceleratorInfo] = None


def detect(*, refresh: bool = False) -> AcceleratorInfo:
    """识别当前后端。进程内缓存——torch build 在进程生命周期里不会变。

    ``refresh=True`` 强制重新探测，仅测试用（真实场景没有热切换后端的路径）。

    torch 未装 / import 失败时返回 ``backend="cpu"`` 且 ``import_error`` 带原因
    ——调用方要区分「真 CPU 机器」与「torch 坏了」时看这个字段。
    """
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE

    try:
        import torch
    except Exception as exc:  # noqa: BLE001  torch 未装 / DLL 加载失败都算探测失败
        _CACHE = AcceleratorInfo(
            backend="cpu", import_error=f"{type(exc).__name__}: {exc}"
        )
        return _CACHE

    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    # 判定顺序：hip 优先。DTK wheel 上 version.cuda 恒为 None，不会误撞 cuda 分支；
    # 反过来 NVIDIA wheel 的 version.hip 也恒为 None，两条互斥。
    if hip:
        backend: Backend = "dcu"
    elif cuda:
        backend = "cuda"
    else:
        backend = "cpu"

    available = False
    count = 0
    names: list[str] = []
    arches: list[str] = []
    if backend != "cpu":
        try:
            available = bool(torch.cuda.is_available())
            if available:
                count = int(torch.cuda.device_count())
                for i in range(count):
                    names.append(torch.cuda.get_device_name(i))
                    arch = getattr(torch.cuda.get_device_properties(i), "gcnArchName", None)
                    if arch:
                        arches.append(str(arch))
        except Exception as exc:  # noqa: BLE001  驱动 / 设备节点问题不该让检测崩
            logger.debug("加速器设备枚举失败: %s", exc)

    _CACHE = AcceleratorInfo(
        backend=backend,
        torch_version=getattr(torch, "__version__", None),
        cuda_version=cuda,
        hip_version=hip,
        available=available,
        device_count=count,
        device_names=tuple(names),
        gcn_arch=tuple(arches),
    )
    return _CACHE


def backend() -> Backend:
    """``detect().backend`` 的快捷式。"""
    return detect().backend


def is_dcu() -> bool:
    """当前是海光 DCU（DTK / HIP build）。"""
    return detect().backend == "dcu"


def is_nvidia() -> bool:
    """当前是 NVIDIA CUDA build。"""
    return detect().backend == "cuda"


# ---------------------------------------------------------------------------
# smi 工具（bootstrap 阶段可用——不依赖 torch）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareProbe:
    """**不依赖 torch** 的硬件探测结果，给 venv 首装等 bootstrap 场景用。

    此时 venv 里往往只有 pip，torch 还没装，无法用 :func:`detect`；只能靠机器上
    有哪个 smi 工具反推该装什么 wheel。
    """

    backend: Backend
    #: 驱动版本（NVIDIA 形如 ``"550.54"``；DCU 侧从 hy-smi 解析，拿不到则 None）
    driver_version: Optional[str] = None
    gpu_name: Optional[str] = None
    #: 实际命中的 smi 可执行文件路径
    smi_path: Optional[str] = None


def _run(args: list[str], timeout: int = 10) -> Optional[str]:
    """跑外部命令拿 stdout；不存在 / 失败 / 超时都返回 None。

    ``errors="replace"`` 是必需的：Windows 中文 locale (cp936) 下解码 smi 输出
    可能抛 UnicodeDecodeError，而探测失败不该让调用方崩。
    """
    exe = shutil.which(args[0])
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, *args[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("%s 执行失败: %s", args[0], exc)
        return None
    if out.returncode != 0:
        return None
    return (out.stdout or "").strip()


def smi_command() -> Optional[str]:
    """当前后端的 smi 可执行文件路径；找不到返回 None。"""
    for cand in _SMI_CANDIDATES[backend()]:
        path = shutil.which(cand)
        if path:
            return path
    return None


def probe_stdlib() -> HardwareProbe:
    """**纯 stdlib** 硬件探测：先试 nvidia-smi，再试海光工具。

    每次调用都真跑 smi（不缓存）——bootstrap 阶段调用次数是个位数，而缓存会让
    「装完驱动再重试」的场景拿到过期结果。

    DCU 判定用两个信号，任一命中即可：
    1. ``hy-smi`` / ``rocm-smi`` 在 PATH 上
    2. ``/dev/kfd`` 存在——HSA kernel driver 的设备节点。DTK 容器忘挂 ``--device=/dev/kfd``
       时 smi 也会失效，但节点检测能区分「没这个硬件」与「容器没挂进来」
    """
    nv = _run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"])
    if nv:
        parts = [p.strip() for p in nv.splitlines()[0].split(",", 1)]
        return HardwareProbe(
            backend="cuda",
            driver_version=parts[0] if parts else None,
            gpu_name=parts[1] if len(parts) > 1 else None,
            smi_path=shutil.which("nvidia-smi"),
        )

    for tool in ("hy-smi", "rocm-smi"):
        path = shutil.which(tool)
        if not path:
            continue
        return HardwareProbe(
            backend="dcu",
            driver_version=_parse_dcu_driver_version(_run([tool, "--version"]) or ""),
            gpu_name=_parse_dcu_gpu_name(_run([tool]) or ""),
            smi_path=path,
        )

    if os.path.exists("/dev/kfd"):
        # 有 HSA 设备节点但没 smi 工具：DTK 装得不全，仍按 DCU 处理（torch 侧
        # 大概率照样能用；报错文案会引导用户查 DTK 安装）。
        return HardwareProbe(backend="dcu")

    return HardwareProbe(backend="cpu")


#: hy-smi / rocm-smi 的 ``--version`` 输出里第一个 ``x.y[.z]`` 版本号。
#: 两家格式都形如 ``ROCM-SMI version: 1.4.1`` / ``HY-SMI version: ...``，取第一个即可。
_VERSION_RE = re.compile(r"(\d+\.\d+(?:\.\d+)*)")


def _parse_dcu_driver_version(text: str) -> Optional[str]:
    """从 smi ``--version`` 输出里抠版本号；抠不到返回 None。

    容错优先：DCU 侧驱动版本**只用于展示与排错**，不参与任何 wheel 选择决策
    （DTK torch 由镜像预装，见 :func:`should_manage_torch_install`），所以解析
    失败无害，不值得为它做严格 schema。
    """
    m = _VERSION_RE.search(text or "")
    return m.group(1) if m else None


def _parse_dcu_gpu_name(text: str) -> Optional[str]:
    """从 smi 输出里找卡型号（形如 ``BW1000`` / ``Z100L`` / ``K100_AI``）。

    与驱动版本同理，仅用于展示；权威型号名来自 ``torch.cuda.get_device_name()``
    （见 :func:`detect`），本函数只服务 torch 还没装好的 bootstrap 阶段。

    海光型号命名是「字母前缀 + 可选 3-4 位数字 + 可选后缀」。数字**可选**是真机
    要求：BW1000 上 ``hy-smi`` 的表格里根本没有型号列（列是 HCU / Temp / AvgPwr /
    ... / Mode），而 ``torch.cuda.get_device_name()`` 返回的就是裸 ``"BW"``。
    早先要求必须有 3-4 位数字的写法在真机上恒返回 None。

    数字可选带来的误匹配风险由「必须整词 + 前缀白名单」压住：只认 BW / Z / K /
    DCU 四个前缀，且两侧是词边界。表头里的 ``HCU`` 不含这些前缀，不会误命中。
    """
    for line in (text or "").splitlines():
        m = re.search(r"\b((?:BW|DCU|Z|K)\s?\d{0,4}[A-Za-z_]*)\b", line)
        if m:
            token = m.group(1).strip()
            # 光一个前缀字母（``Z`` / ``K``）几乎肯定是别的东西误命中（表头缩写等）；
            # 多字母前缀（BW / DCU）单独出现是合法型号名，真机就是这样。
            if len(token) > 1:
                return token
    return None


# ---------------------------------------------------------------------------
# 装包策略（各 runtime service 问这里，不要自己按后端 if）
# ---------------------------------------------------------------------------


def should_manage_torch_install() -> bool:
    """本项目是否该替用户装 / 重装 torch。

    **DCU 上必须是 False**，这是本次移植最关键的一条护栏：DTK torch 由厂商镜像
    预装（wheel 不在 PyPI 上，且与镜像内的 DTK 运行时严格配套）。任何
    ``pip install torch`` 都会把它换成 PyPI 的 CPU 版或 NVIDIA 版，环境直接报废
    且无法用 pip 装回来——用户只能重建容器。

    所以 DCU 上：``studio.sh`` 首装跳过选 wheel、Settings 的「重装 PyTorch」置灰、
    CLI 的 ``--torch=<tag>`` 拒绝执行。
    """
    return backend() != "dcu"


def supports_prebuilt_flash_attn_wheels() -> bool:
    """能否用 ``flash_attention.py`` 那套 GitHub prebuilt wheel。

    那些 wheel 文件名里带 ``cu126`` / ``cu130`` 这类 CUDA 标签、链接的是 CUDA
    runtime，DCU 上装不了也用不了。海光的 flash-attn 走 DTK 自家渠道单独发布，
    与 GitHub 上的 CUDA wheel 不是同一条供应链，不能混用。
    """
    return backend() == "cuda"


def can_pip_install_xformers() -> bool:
    """能否用 ``pip install xformers`` 从公开源装到可用的 wheel。

    **只回答「能不能自动装」，不回答「装了能不能用」** —— 这两件事在 DCU 上不一致，
    早期实现把它们混为一谈是个错误（详见 :func:`xformers_works`）。

    上游 xformers 只发 CUDA build（PyTorch 官方 index 的 cuXXX 分组下），ROCm / DTK
    侧公开源上没有对应 wheel。海光在光合社区单独发布配套 wheel（如
    ``xformers-0.0.33+das.opt1.dtk2604.torch251``），但那不在 PyPI 上、也无法靠
    ``--index-url`` 找到，只能手动 pip install。
    """
    return backend() == "cuda"


def xformers_works() -> bool:
    """xformers 在当前环境**实际可用**（能 import 且能真的算出结果）。进程内缓存。

    与 :func:`can_pip_install_xformers` 的区别是本次移植的一个重要教训：最初把
    「DCU 上 xformers 不可用」当成硬事实写进能力矩阵，而海光其实**有**配套 wheel
    （光合社区发布，与 DTK / torch 版本严格配套）。装上之后 NaViT 打包等依赖 xformers
    的功能在 DCU 上照样能跑，所以判据必须是「实测能不能用」而不是「是什么卡」。

    实测而非只 import 的理由与 :func:`probe_sdpa_backends` 同源：海光的 xformers
    wheel 是 ``py3-none-any``（纯 Python），底层 kernel 依赖 DTK 侧的 flash-attn /
    HIP 实现 —— import 成功不代表 ``memory_efficient_attention`` 真能跑。
    """
    global _XFORMERS_CACHE
    if _XFORMERS_CACHE is not None:
        return _XFORMERS_CACHE

    ok = False
    try:
        import torch
        import xformers.ops as xops

        if torch.cuda.is_available():
            q = torch.randn(1, 128, 2, 64, device="cuda", dtype=torch.bfloat16)
            xops.memory_efficient_attention(q, q, q)
            del q
            ok = True
        else:
            # 没设备时无法实测。按「能 import 就算可用」处理 —— 这条路径只在
            # CPU 机器上出现（不训练），保守放行不会有实际后果。
            ok = True
    except Exception as exc:  # noqa: BLE001  未装 / ABI 不符 / kernel 缺失都算不可用
        logger.debug("xformers 不可用: %s", exc)

    _XFORMERS_CACHE = ok
    return ok


def onnx_gpu_provider() -> Optional[str]:
    """当前后端对应的 onnxruntime GPU ExecutionProvider 名。

    ``None`` 表示该后端没有可用的 GPU EP，打标只能跑 CPU。DCU 侧海光提供
    MIGraphX EP（需装 DTK 配套的 onnxruntime，不是 PyPI 的 onnxruntime-gpu）。
    """
    return {
        "cuda": "CUDAExecutionProvider",
        "dcu": "MIGraphXExecutionProvider",
        "cpu": None,
    }[backend()]


# ---------------------------------------------------------------------------
# 显存查询（替代直接用 pynvml 的调用点）
# ---------------------------------------------------------------------------


def free_vram_bytes() -> Optional[int]:
    """全卡真实空闲显存（0 号卡）；查询失败返回 None。

    两条路径按后端选，**顺序有讲究**：

    - NVIDIA：优先 NVML。Windows WDDM 下 ``cudaMemGetInfo`` 是**每进程虚拟化
      视角**，看不到其他进程占用（真机实测：他进程持有 20GB 时它仍报全量 free），
      用它做跨进程护栏形同虚设；NVML 是全卡视角。
    - DCU：走 torch ``mem_get_info``。DCU 只在 Linux 上，没有 WDDM 那套虚拟化，
      ``mem_get_info`` 本身就是全卡视角；且 pynvml 在 DCU 上根本 init 不了。

    调用方（``runtime/training/sysmem.py`` 的水位护栏）拿到 None 时静默放行。
    """
    if backend() == "cuda":
        nvml = _nvml_free_bytes()
        if nvml is not None:
            return nvml
    return _torch_free_bytes()


def _nvml_free_bytes() -> Optional[int]:
    """NVML 视角的 0 号卡空闲显存；NVIDIA 专用。"""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return int(pynvml.nvmlDeviceGetMemoryInfo(handle).free)
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001  未装 / 无驱动 / 非 NVIDIA 都走 fallback
        return None


def _torch_free_bytes() -> Optional[int]:
    """torch ``mem_get_info`` 视角的空闲显存。"""
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.mem_get_info()[0])
    except Exception:  # noqa: BLE001
        pass
    return None


@dataclass(frozen=True)
class DeviceStats:
    """单卡实时指标。除显存外都可能为 None。

    显存是**必有**字段（torch ``mem_get_info`` 在两个后端上都可靠）；其余是
    best-effort，拿不到就留 None，topbar 按可缺失渲染：

    - ``util_pct`` / ``temp_c``：NVIDIA 走 NVML；DCU 解析 hy-smi 文本。
    - ``power_w`` / ``sclk_mhz``：**只有 DCU 有**，走 sysfs（见
      :func:`_sysfs_power_clock`）。NVIDIA 侧 NVML 能给功率但本项目暂未接，
      留 None。

    ``sclk_mhz`` 单看没有意义（不知道满频是多少），要配 ``sclk_max_mhz`` 一起看 ——
    「当前 1500 / 最高 1500」才说明没降频。判断卡有没有被限制时这一对比功率可靠：
    撞功率墙或温度墙的卡会主动降档，而满频且功率有余量说明瓶颈在算力本身。
    """

    index: int
    name: str
    vram_used_gb: float
    vram_total_gb: float
    util_pct: Optional[int] = None
    temp_c: Optional[int] = None
    power_w: Optional[int] = None
    power_cap_w: Optional[int] = None
    sclk_mhz: Optional[int] = None
    sclk_max_mhz: Optional[int] = None


def device_stats() -> Optional[list[DeviceStats]]:
    """逐卡实时指标。

    返回值三态，调用方（topbar SSE 采样）按此渲染：
    - ``None``：查不到（无加速器 / torch 不可用）→ 前端隐藏 GPU pill
    - ``[]``：后端可用但 0 卡
    - 非空列表：正常数据

    NVIDIA 优先 NVML（利用率 + 温度齐全），拿不到时退到 torch；DCU 直接走 torch。
    """
    if backend() == "cuda":
        nvml = _nvml_device_stats()
        if nvml is not None:
            return nvml
    return _torch_device_stats()


def _nvml_device_stats() -> Optional[list[DeviceStats]]:
    """NVML 逐卡指标（利用率 / 温度齐全）；失败返回 None 让调用方 fallback。"""
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001
        logger.debug("pynvml 不可用: %s", exc)
        return None
    try:
        out: list[DeviceStats] = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                util = int(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            except Exception:  # noqa: BLE001
                util = None
            try:
                temp = int(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:  # noqa: BLE001
                temp = None
            out.append(DeviceStats(
                index=i,
                name=name,
                vram_used_gb=round(mem.used / 1024**3, 2),
                vram_total_gb=round(mem.total / 1024**3, 2),
                util_pct=util,
                temp_c=temp,
            ))
        return out
    except Exception:  # noqa: BLE001
        logger.debug("NVML 逐卡采集失败", exc_info=True)
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# SDPA 后端可用性（真机实测，非静态推断）
# ---------------------------------------------------------------------------

#: SDPA 后端名 → 是否实测可用。``configure_sdpa()`` 首次调用后填充。
_SDPA_CACHE: Optional[dict[str, bool]] = None

#: xformers 是否实测可用（:func:`xformers_works` 的缓存）。
_XFORMERS_CACHE: Optional[bool] = None

#: 探测用的最小 q/k/v 形状 [B, H, S, D]。S=128 要够大 —— flash / mem-efficient
#: kernel 对极小序列可能直接判定不适用而回落，那样探测结果会假阴性。
_SDPA_PROBE_SHAPE = (1, 2, 128, 64)


def probe_sdpa_backends() -> dict[str, bool]:
    """实测每个 SDPA 后端能否真的算出结果。返回 ``{"flash": bool, "mem_efficient": bool,
    "math": bool, "default": bool}``；进程内缓存。

    **为什么必须实测而不是查标志位**：海光 DTK 的 torch 编译时**开启**了 flash 后端，
    但把 kernel 委托给外部 ``flash_attn_2_cuda*.so``（海光把 flash-attn 作为独立包
    发布）。包没装时 ``torch.backends.cuda.flash_sdp_enabled()`` 仍报 True，而真正
    调用会抛 ``RuntimeError: No matching libraries found for flash_attn_2_cuda*.so``。
    真机实测（BW / gfx936 / DTK 26.04 / torch 2.9.0）：

        sdpa_default        → RuntimeError: No matching libraries ... flash_attn_2_cuda*.so
        sdpa_flash          → 同上
        sdpa_mem_efficient  → RuntimeError: No available kernel. Aborting execution.
        sdpa_math           → OK

    注意 ``default`` 也失败：SDPA 的默认 dispatch 会**先试 flash**，撞上这个异常就
    直接抛出，不会优雅回落到 math。所以「什么都不配置、依赖 SDPA 自己选」在这台机器上
    等于训练第一个 attention 调用就崩 —— 这正是本函数存在的理由。

    探测代价：4 次小张量 SDPA（形状 [1,2,128,64]），~1MB 显存，一次性。
    """
    global _SDPA_CACHE
    if _SDPA_CACHE is not None:
        return _SDPA_CACHE

    result = {"flash": False, "mem_efficient": False, "math": False, "default": False}
    try:
        import warnings

        import torch
        import torch.nn.functional as F

        if not torch.cuda.is_available():
            _SDPA_CACHE = result
            return result

        # 整段静音 UserWarning。本函数**故意**去调用注定失败的后端，torch 于是逐条
        # 抱怨（"Flash attention kernel not used because…"、"Torch was not compiled
        # with cuDNN attention"…）。这些警告对探测本身是预期结果、不是问题，但直接
        # 喷到训练日志里会让用户以为环境坏了 —— 真机上一次探测刷了 8 行。
        # 探测结论该怎么告知用户由 configure_sdpa() 统一负责（分级 warn / debug）。
        #
        # 只包探测这一段：出了这个 with，后续训练里真正的 attention 警告照常可见。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            probe = torch.randn(*_SDPA_PROBE_SHAPE, device="cuda", dtype=torch.bfloat16)

            def _works(q, ctx=None) -> bool:
                """跑一次 SDPA，成功即该后端可用。``ctx=None`` 表示测默认 dispatch。"""
                try:
                    if ctx is None:
                        F.scaled_dot_product_attention(q, q, q)
                    else:
                        with ctx:
                            F.scaled_dot_product_attention(q, q, q)
                    return True
                except Exception:  # noqa: BLE001  探测失败即「不可用」，这是本函数的语义
                    return False

            from torch.nn.attention import SDPBackend, sdpa_kernel

            result["default"] = _works(probe)
            for name, backend in (
                ("flash", SDPBackend.FLASH_ATTENTION),
                ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
                ("math", SDPBackend.MATH),
            ):
                result[name] = _works(probe, sdpa_kernel(backend))
            del probe
    except Exception as exc:  # noqa: BLE001  torch 太老没 sdpa_kernel / 设备问题
        logger.debug("SDPA 后端探测失败: %s", exc)

    _SDPA_CACHE = result
    return result


def configure_sdpa() -> dict[str, bool]:
    """按实测结果关掉不可用的 SDPA 后端，返回探测结果。

    在训练 / 推理启动期调一次。**只在 DCU 上生效**：NVIDIA 路径一行不改，避免为这条
    护栏引入任何既有行为变化（NVIDIA 上 SDPA 的 dispatch 与回落一向正常）。

    做法是设 ``torch.backends.cuda.enable_*_sdp`` 全局开关，把探测中失败的后端关掉，
    让 SDPA 的 dispatch 根本不会去试它们。这比在每个调用点包 ``sdpa_kernel(MATH)``
    好：调用点散落在 modeling 各处（cosmos / krea2 / VAE / text encoder），全局开关
    一次覆盖，且用户后来装上海光 flash-attn 时探测会自动放行、无需改代码。

    代价说明：只剩 math 后端时 attention 走「显式materialize 完整 [B,H,S,S] 矩阵」，
    比 flash 慢且吃显存（长序列尤甚）。这是**能跑**与**跑得快**之间的选择 —— 装上
    海光渠道的 flash-attn 后本函数会自动恢复 flash 路径。
    """
    probed = probe_sdpa_backends()
    if backend() != "dcu":
        return probed
    try:
        import torch

        cuda_backends = torch.backends.cuda
        # math 永远开着：它是唯一无外部依赖的纯 ATen 实现，也是最后的保底。
        # 万一探测显示 math 也不可用，关掉其余后端只会让报错更早更清楚，不会更糟。
        cuda_backends.enable_math_sdp(True)
        if not probed["flash"]:
            cuda_backends.enable_flash_sdp(False)
        if not probed["mem_efficient"]:
            cuda_backends.enable_mem_efficient_sdp(False)
        # cuDNN attention 后端：DTK 上真机报 "Torch was not compiled with cuDNN
        # attention"，探测里没单独测（SDPBackend.CUDNN_ATTENTION 在部分 torch 版本
        # 不存在），直接关掉最省事 —— HIP 上它不可能有实现。
        disable_cudnn = getattr(cuda_backends, "enable_cudnn_sdp", None)
        if disable_cudnn is not None:
            disable_cudnn(False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("配置 SDPA 后端失败: %s", exc)
        return probed

    # 分级报告：flash 不可用才是**性能问题**（要 warn 并给出可执行的解决办法）；
    # 只是 mem_efficient 缺失则无所谓（flash 更快，本就优先它），debug 记一笔即可。
    # 真机实测（BW1000 / DTK 26.04 / torch 2.5.1 + 海光 flash-attn 2.6.1）：flash 与
    # math 可用、mem_efficient 恒报 "No available kernel"（DTK 编译时未开该后端），
    # 所以「只缺 mem_efficient」是这个平台的**正常终态**，不该每次启动都刷警告。
    if not probed["flash"]:
        logger.warning(
            "%s 上 SDPA 的 flash 后端不可用，已关闭，attention 走 math 后端"
            "（能跑但更慢、长序列更吃显存）。"
            "flash 后端依赖海光单独发布的 flash-attn 包（torch 内部 dlopen "
            "flash_attn_2_cuda*.so）—— 从光合开发者社区取与镜像 DTK / torch 版本匹配的"
            "wheel 手动 pip install 后，本项目会自动改走 flash。"
            "先跑 `bash tools/find_flash_attn.sh` 看本机有没有现成包。",
            detect().vendor_label,
        )
    elif not probed["mem_efficient"]:
        logger.debug(
            "SDPA mem-efficient 后端不可用（已关闭）；flash 可用，走 flash 即最优路径"
        )
    return probed


def usable_sdpa_backends():
    """实测可用的 ``SDPBackend`` 优先级列表（快的在前）；torch 太老返回 None。

    给需要 ``sdpa_kernel(priority, set_priority=True)`` 的调用点用 —— 那种写法会
    **覆盖**全局开关，所以不能让它盲目列全部后端（DCU 上会白试一遍 flash 再靠
    except 兜回来，每次调用都刷一堆 warning）。
    """
    probed = probe_sdpa_backends()
    try:
        from torch.nn.attention import SDPBackend
    except Exception:  # noqa: BLE001
        return None
    order = [
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    ]
    usable = [b for name, b in order if probed.get(name)]
    # 全都探测失败（极端情况）时返回 MATH 兜底：让调用点仍有个合法列表可用，
    # 真不行的话报错发生在 SDPA 内部，信息比这里静默返回 None 更有用。
    return usable or [SDPBackend.MATH]


#: ``hy-smi`` / ``rocm-smi`` 表格数据行。真机（DTK 26.04 / BW1000）格式：
#:
#:     HCU     Temp     AvgPwr     Perf     PwrCap     VRAM%      HCU%      Dec%   Enc%   Mode
#:     0       50.0C    80.0W      auto     1000.0W    0%         0.0%      0.0%   0.0%   Normal
#:
#: 只抓需要的三列：卡号、温度、HCU%（= GPU 利用率）。中间几列用宽松的 ``\S+`` 跳过，
#: 避免 DTK 版本间增删列就整条失配。
#:
#: 刻意不抓另外两列：
#: - ``VRAM%``：是百分比，而 topbar 要显示绝对值，torch ``mem_get_info`` 更准更细。
#: - ``AvgPwr``：**实测不可信**。训练满载同一时刻 hy-smi 报 79 W / 95 W，而 sysfs
#:   ``power1_average`` 是 564 W / 563 W。功率走 :func:`_sysfs_card_metrics`。
_HY_SMI_ROW = re.compile(
    r"^\s*(\d+)\s+([\d.]+)C\s+\S+\s+\S+\s+\S+\s+\S+%\s+([\d.]+)%",
)


def _parse_hy_smi_metrics(text: str) -> dict[int, tuple[Optional[int], Optional[int]]]:
    """解析 smi 表格 → ``{卡号: (util_pct, temp_c)}``；解析不出的行跳过。

    best-effort：这两项是 topbar 的可选渲染项，拿不到就不显示（前端已按可缺失处理）。
    宁可返回空 dict 也不抛 —— 调用方是 2-3s 轮询的热路径。
    """
    out: dict[int, tuple[Optional[int], Optional[int]]] = {}
    for line in (text or "").splitlines():
        m = _HY_SMI_ROW.match(line)
        if not m:
            continue
        try:
            idx = int(m.group(1))
            temp = int(round(float(m.group(2))))
            util = int(round(float(m.group(3))))
        except ValueError:
            continue
        out[idx] = (util, temp)
    return out


def _dcu_smi_metrics() -> dict[int, tuple[Optional[int], Optional[int]]]:
    """跑 smi 拿逐卡 (利用率, 温度)；不可用返回空 dict。"""
    smi = smi_command()
    if not smi:
        return {}
    return _parse_hy_smi_metrics(_run([smi]) or "")


#: DRM sysfs 根目录。模块级常量是为了测试能 monkeypatch 到 tmp_path —— 逐卡功率
#: 这条路径必须能在没有 DCU 的机器上测。
_DRM_ROOT = "/sys/class/drm"

#: 容器可见的 DRM 设备节点目录。``/dev/dri/cardN`` 是内核给本容器暴露的节点，
#: 与 hy-smi 无关 —— 后者在容器里会枚举到宿主机的全部卡。
_DRI_ROOT = "/dev/dri"


def _read_sysfs_int(path: str) -> Optional[int]:
    """读一个整数 sysfs 文件。任何异常（不存在 / 权限 / 非数字）返回 None。

    热路径（topbar 2-3s 轮询）上不抛异常，也不打日志 —— 缺文件是常态
    （不同 DTK 版本暴露的字段不同）。
    """
    try:
        with open(path, encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _parse_dpm_sclk(text: str) -> tuple[Optional[int], Optional[int]]:
    """解析 ``pp_dpm_sclk`` → (当前频率 MHz, 最高档频率 MHz)。

    真机格式（DTK 26.04 / BW1000，每行一档，``*`` 标当前档）::

        0: 300Mhz
        1: 600Mhz
        ...
        10: 1500Mhz *

    返回「当前」和「最高」两个值：单独一个当前频率没法判断有没有降频，
    得知道满频是多少才有意义。
    """
    cur: Optional[int] = None
    mx: Optional[int] = None
    for line in (text or "").splitlines():
        m = re.search(r"(\d+)\s*Mhz", line, re.IGNORECASE)
        if not m:
            continue
        try:
            mhz = int(m.group(1))
        except ValueError:
            continue
        if mx is None or mhz > mx:
            mx = mhz
        if "*" in line:
            cur = mhz
    return cur, mx


def visible_drm_cards() -> list[int]:
    """本容器可见的 DRM 卡号，数字升序。真机上是 ``[6, 8]``（宿主机有 8+ 张）。

    为什么用 ``/dev/dri`` 而不是 hy-smi 的卡号：容器里 hy-smi **会枚举到宿主机
    的全部卡**，而 ``/dev/dri`` 只有内核分给本容器的节点 —— 这是设备节点级的隔离，
    比任何工具的报数都可靠。

    卡号**不固定**：容器重建后可能从 [6, 8] 变成别的，所以每次读、不跨调用缓存。
    """
    try:
        nums = [
            name[4:] for name in os.listdir(_DRI_ROOT)
            if name.startswith("card") and name[4:].isdigit()
        ]
    except OSError:
        return []
    return sorted(int(n) for n in nums)


def _sysfs_card_metrics(card_no: int) -> dict[str, Optional[int]]:
    """单张 DRM 卡的 (功率 W, 功率上限 W, 当前频率 MHz, 最高频率 MHz)。

    **为什么功率不取 hy-smi 的 AvgPwr 列**：真机实测（DTK 26.04 / BW1000，训练满载
    同一时刻）——

        hy-smi AvgPwr :  79 W /  95 W
        sysfs power1_average : 564 W / 563 W

    差 6-7 倍。而同一份 hy-smi 输出里 ``VRAM%`` / ``HCU%`` / ``Temp`` 三列都与 sysfs
    吻合（83% vs 53/63GiB、98.8% vs busy 100%、70C 介于核心 58C 与热点 75C 之间），
    所以**不是读错了卡**，是 AvgPwr 这一列本身不可信。sysfs 的值可交叉验证：满载
    577-580 W、同机空闲卡 76-89 W，区分度正常。
    """
    d = f"{_DRM_ROOT}/card{card_no}/device"
    out: dict[str, Optional[int]] = {
        "power_w": None, "power_cap_w": None,
        "sclk_mhz": None, "sclk_max_mhz": None,
    }
    # hwmon 目录名（hwmon7 / hwmon13 ...）随卡而异，遍历取第一个有 power1_average 的。
    try:
        hwmons = sorted(os.listdir(f"{d}/hwmon"))
    except OSError:
        hwmons = []
    for h in hwmons:
        uw = _read_sysfs_int(f"{d}/hwmon/{h}/power1_average")
        if uw is None:
            continue
        out["power_w"] = int(round(uw / 1_000_000))  # 微瓦 → 瓦
        cap = _read_sysfs_int(f"{d}/hwmon/{h}/power1_cap_max")
        if cap is None:
            cap = _read_sysfs_int(f"{d}/hwmon/{h}/power1_cap")
        if cap is not None:
            out["power_cap_w"] = int(round(cap / 1_000_000))
        break
    try:
        with open(f"{d}/pp_dpm_sclk", encoding="utf-8") as f:
            out["sclk_mhz"], out["sclk_max_mhz"] = _parse_dpm_sclk(f.read())
    except OSError:
        pass
    return out


def _dcu_sysfs_metrics(device_count: int) -> dict[int, dict[str, Optional[int]]]:
    """DCU 逐卡 sysfs 指标，键是 **torch 序号**。拿不到映射时返回空 dict。

    映射依据：``/dev/dri`` 里可见卡号按数字升序 ↔ torch 序号 0..N-1。这个对应关系
    有实测支撑 —— hy-smi 的 ``VRAM%`` / ``HCU%`` / ``Temp`` 三列与按此映射读到的
    sysfs 值一致（见 :func:`_sysfs_card_metrics` 的数据）。

    可见卡数与 ``device_count`` 不一致时**整体放弃**而不是部分填充：宁可前端不显示
    功率，也不能把 A 卡的功率标到 B 卡上 —— 那种错误看不出来，比缺失有害得多。
    """
    cards = visible_drm_cards()
    if not cards or len(cards) != device_count:
        if cards:
            logger.debug(
                "DRM 可见卡数 %d != torch device_count %d，跳过 sysfs 功率采集"
                "（避免卡号错位把功率标到错的卡上）", len(cards), device_count,
            )
        return {}
    return {i: _sysfs_card_metrics(no) for i, no in enumerate(cards)}


def _torch_device_stats() -> Optional[list[DeviceStats]]:
    """torch 视角逐卡指标。三个数据源各管一段，都是 best-effort。

    DCU 的主路径。分工的理由：
    - **显存**用 torch ``mem_get_info`` —— 绝对值、按卡精确，而 smi 只给 ``VRAM%``。
    - **利用率 / 温度**用 hy-smi —— torch 压根不暴露这两项，而 smi 这两列实测可信。
    - **功率 / 频率**用 sysfs —— hy-smi 的 ``AvgPwr`` 列实测偏差 6-7 倍
      （见 :func:`_sysfs_card_metrics`），频率它压根不给。

    三段都可以缺：解析失败 / 工具不存在 / 卡号映射对不上时相应字段留 None，
    显存照常返回（前端按可缺失渲染）。NVIDIA 走不到这里（NVML 路径优先，见
    :func:`device_stats`），所以不必担心多跑一次 nvidia-smi。
    """
    info = detect()
    if not info.is_gpu:
        return None
    is_dcu = info.backend == "dcu"
    metrics = _dcu_smi_metrics() if is_dcu else {}
    sysfs = _dcu_sysfs_metrics(info.device_count) if is_dcu else {}
    try:
        import torch

        out: list[DeviceStats] = []
        for i in range(info.device_count):
            free, total = torch.cuda.mem_get_info(i)
            util, temp = metrics.get(i, (None, None))
            sf = sysfs.get(i, {})
            out.append(DeviceStats(
                index=i,
                name=info.device_names[i] if i < len(info.device_names) else "?",
                vram_used_gb=round((total - free) / 1024**3, 2),
                vram_total_gb=round(total / 1024**3, 2),
                util_pct=util,
                temp_c=temp,
                power_w=sf.get("power_w"),
                power_cap_w=sf.get("power_cap_w"),
                sclk_mhz=sf.get("sclk_mhz"),
                sclk_max_mhz=sf.get("sclk_max_mhz"),
            ))
        return out
    except Exception:  # noqa: BLE001
        logger.debug("torch 逐卡采集失败", exc_info=True)
        return None
