"""onnxruntime 运行时检测 / 安装（Settings 页驱动）。

onnxruntime（CPU）和 onnxruntime-gpu 共享 import 名且**互斥**，不能同装；
不同机器 CUDA 版本不同，requirements.txt 不写死它。安装由用户在 Settings →
ONNX Runtime 主动触发，对齐 xformers / flash-attention 的模式 —— 不用打标的
用户不会被启动期 pip install 阻塞。

主路径：
    install_runtime(target) — Settings 页「安装 / 重装为 X」按钮调；同步 pip
    current_runtime()       — Settings 页展示当前状态 / cli.py 启动期检查
    detect_cuda()           — nvidia-smi 探针

约定：
- onnxruntime-gpu 的版本约束按 **CUDA 大版本**（cu12/cu13）分流，锚定 _resolve_cuda_major()
  （= torch.version.cuda 的 major）：ORT build 必须与 torch 同 major，否则同进程
  cu12/cu13 混用 ABI 错位（ORT 1.26+ 默认 CUDA 13，旧 `>=1.20` 会拉到它而崩
  libcudart.so.13）。cu12 钉 `>=1.20,<1.26`、cu13 用 `>=1.26`；预加载 soname + 补的
  runtime wheel 也按同一 major 选 cu12（nvidia-*-cu12）/ cu13（去后缀名）两套。
- 「装错了 cuda 版本」体现为 `import onnxruntime` 成功但 `CUDAExecutionProvider`
  不在 providers 里；不自动重装（用户可能故意），UI 给手动按钮 + 警告
- 装包用 `subprocess.run([sys.executable, "-m", "pip", ...])`，不调内部 pip API

PP9.5 — CUDA 共享库预加载 + session 创建 fallback：
- onnxruntime-gpu 不带 CUDA runtime so（libcurand / libcublas / libcudnn ...）。
  Linux 上常见踩坑：`get_available_providers()` 报 CUDA EP 可用，但创 session
  时 dlopen 挂在 `libcurand.so.10: cannot open shared object file`。
- 解法：模块顶层在 import onnxruntime **之前**用 ctypes RTLD_GLOBAL 预加载
  torch 自带的 `nvidia/*/lib/*.so`（PyTorch 默认安装 nvidia-* wheel 到这里）。
  这条路在 ComfyUI / WD14 生态里**没人做**，但是最便宜的通用 fix。
- 失败时 wd14_tagger 仍会捕异常降 CPU；本模块用 record_cuda_load_error 把
  原因 stash 出来给 UI 显示。

海光 DCU（DTK）—— 本模块受影响最大的三处，全部按 utils.accelerator 分派：
- **预加载整段跳过**：DCU 上没有 `site-packages/nvidia/*` 这些包（那是 NVIDIA
  torch wheel 的依赖），而 DTK 配套的 onnxruntime 自己链 DTK 运行时，不需要任何
  外部 preload。返回 dict 里加 `backend_skip=True` 与现有几个 *_skip 同构。
- **GPU 包不是 onnxruntime-gpu**：那是 CUDA build。海光的 GPU EP 是 MIGraphX，
  只在 DTK 配套的 onnxruntime 里，不在 PyPI 上 —— 所以 DCU 上 `gpu` 目标拒绝执行，
  `auto` 装 CPU 包（见 _decide_target 里的决策说明）。
- **GPU EP 名不是 CUDAExecutionProvider**：`current_runtime().cuda_available` 改成
  「当前后端的 GPU EP 可用」，实际 EP 名新增 `gpu_provider` 字段透出。
"""
from __future__ import annotations

import ctypes
import importlib
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Optional

from utils import accelerator

logger = logging.getLogger(__name__)

GPU_PACKAGE = "onnxruntime-gpu"
CPU_PACKAGE = "onnxruntime"
# onnxruntime-directml 给 Windows 用户用 DX12 后端，绕开 CUDA/cuDNN ABI 兼容性
# 问题（issue #231：RTX 5090 + onnxruntime-gpu 静默降 CPU）。任何支持 DX12 的 GPU
# 都能跑（NVIDIA / AMD / Intel）。仅 Windows 有 PyPI wheel。
DIRECTML_PACKAGE = "onnxruntime-directml"
CPU_VERSION_SPEC = ">=1.16"
DIRECTML_VERSION_SPEC = ">=1.20"
# 三包共享 import 名 `onnxruntime`，**互斥**；切换前必须 uninstall 其余两个
_MUTUALLY_EXCLUSIVE_PACKAGES: tuple[str, ...] = (GPU_PACKAGE, CPU_PACKAGE, DIRECTML_PACKAGE)

# onnxruntime-gpu 的版本约束按 **CUDA 大版本**分流 —— ORT build 的 CUDA 大版本必须
# 跟 torch 一致（同进程 cu12/cu13 混用 ABI 错位），所以锚定 _resolve_cuda_major()。
# ORT 1.26.0 起 PyPI 默认 wheel 切到 CUDA 13、1.27.0 彻底删掉 CUDA 12 build；旧的
# `>=1.20` 现在会被解析成 1.27（CUDA 13），与项目 cu128 torch + cu12 预加载/wheel
# 机器对不上 → `import onnxruntime` 崩 libcudart.so.13（详见 _resolve_cuda_major）。
GPU_VERSION_SPEC_CU12 = ">=1.20,<1.26"  # ORT 1.20–1.25 默认 CUDA 12.x
GPU_VERSION_SPEC_CU13 = ">=1.26,<2.0"   # ORT 1.26+ CUDA 13
# 默认线（torch 拿不到 CUDA 大版本时）：cu12，=旧行为，零回归。
DEFAULT_CUDA_MAJOR = 12

# PP9.6 — onnxruntime-gpu wheel 不打包 CUDA runtime so（libcurand / libcublas
# / ...），用户机器没系统装 CUDA 时 dlopen 直接挂。靠 PyPI 上的 nvidia-* wheel
# 把它们装到 venv 的 site-packages/nvidia/*/lib/，配合本模块顶层的 RTLD_GLOBAL
# preload 让 onnxruntime 后续 dlopen 找到符号。
#
# 注意：
# - **不含 cuDNN wheel**：torch 的 GPU build 已经把它装上 + 锁死，再
#   `pip install nvidia-cudnn-cu12` 不带版本会被升到最新，破坏 torch；只在
#   它**完全没装**时才补。
# - 这套 wheel 只有 manylinux 平台；Windows / macOS 上不可用 → 安装函数会
#   早返回。Windows 上正确路径是用户系统装 CUDA Toolkit + cuDNN。
# CUDA 12 与 CUDA 13 的 nvidia wheel 包名不同：CUDA 13 的 `nvidia-*-cu13` 是 PyPI
# 上废弃的空占位包，真包是去后缀的 `nvidia-cuda-runtime` / `nvidia-cublas` …；
# 只有 cuDNN 仍带 `-cu13`。两套各自按 _resolve_cuda_major() 选用。
_NVIDIA_CUDA_RUNTIME_WHEELS_CU12: tuple[str, ...] = (
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-cusolver-cu12",
)
_NVIDIA_CUDNN_WHEEL_CU12 = "nvidia-cudnn-cu12"
_NVIDIA_CUDA_RUNTIME_WHEELS_CU13: tuple[str, ...] = (
    # cu13 的 runtime/算子库改去后缀名（`-cu13` 是 PyPI 上的空占位包）
    "nvidia-cuda-runtime",
    "nvidia-cuda-nvrtc",
    "nvidia-cublas",
    "nvidia-cufft",
    "nvidia-curand",
    "nvidia-cusparse",
    "nvidia-cusolver",
)
_NVIDIA_CUDNN_WHEEL_CU13 = "nvidia-cudnn-cu13"

# torch 装 GPU build 时拉的 nvidia-* wheel 安装到 site-packages/nvidia/<sub>/lib/。
# 预加载对这些子包的 lib/ 下 glob 所有 lib*.so* 做 RTLD_GLOBAL —— 不再写死 soname，
# 自动适配 cu12（libcudart.so.12 / libcudnn.so.9）与 cu13（.so.13 / libcudnn.so.10）。
# 语义本就该加载「torch 自带的全套 CUDA so」；torch 与 ORT 同 major（由 _decide_target
# 保证），跨大版本永不会混。cublasLt 在 nvidia.cublas 包里，glob 一并带上。
_TORCH_NVIDIA_LIB_PKGS_LINUX: tuple[str, ...] = (
    "nvidia.cuda_runtime",
    "nvidia.cuda_nvrtc",
    "nvidia.cublas",
    "nvidia.cufft",
    "nvidia.curand",
    "nvidia.cusparse",
    "nvidia.cusolver",
    "nvidia.cudnn",
)


def _backend() -> str:
    """当前加速器后端；探测异常兜底成 ``"cuda"``。

    兜底选 cuda 是为了让探测失败时 NVIDIA 路径**逐字节保持旧行为** —— 本模块新增
    的后端分支全部形如「是 dcu 才走新路」，兜底 cuda 等于全部不生效。
    本模块在 import 期就会调它（_ensure_preload），所以绝不能让它抛。
    """
    try:
        return accelerator.backend()
    except Exception:  # noqa: BLE001
        return "cuda"


def _gpu_ep_name() -> str:
    """当前后端的 onnxruntime GPU ExecutionProvider 名。**永不返回 None。**

    只有 DCU 走 ``accelerator.onnx_gpu_provider()``（→ MIGraphX），其余后端一律
    ``CUDAExecutionProvider``。**这里刻意不直接透传 accelerator 的三态**：那个函数
    在 ``backend=="cpu"`` 时返回 None，而「torch 是 CPU build / 没装 torch，但
    venv 里装着 onnxruntime-gpu 且 CUDA EP 可用」是完全合法且常见的状态（打标用
    GPU、训练还没配好），历史上 ``cuda_available`` 在这种机器上就是 True。直接透传
    会把它变成 False —— 一个纯 NVIDIA 侧的行为回归。

    onnxruntime 的 GPU 能力与 torch build 无关，这是两个独立的包；只有「换了厂商」
    才需要换 EP 名。
    """
    if _backend() == "dcu":
        try:
            return accelerator.onnx_gpu_provider() or "MIGraphXExecutionProvider"
        except Exception:  # noqa: BLE001
            return "MIGraphXExecutionProvider"
    return "CUDAExecutionProvider"


def _vendor_label() -> str:
    """面向用户的后端名（报错 / 日志文案用）。"""
    return accelerator.VENDOR_LABEL.get(_backend(), _backend())  # type: ignore[arg-type]


def _cuda_wheels_for(major: Optional[int]) -> tuple[tuple[str, ...], str]:
    """按 CUDA 大版本返回 (runtime_wheels, cudnn_wheel)。None / 12 → cu12；13 → cu13。"""
    if major == 13:
        return _NVIDIA_CUDA_RUNTIME_WHEELS_CU13, _NVIDIA_CUDNN_WHEEL_CU13
    return _NVIDIA_CUDA_RUNTIME_WHEELS_CU12, _NVIDIA_CUDNN_WHEEL_CU12


def _gpu_version_spec_for(major: Optional[int]) -> str:
    """按 CUDA 大版本返回 onnxruntime-gpu 的 pip 版本约束字符串。"""
    if major == 13:
        return f"{GPU_PACKAGE}{GPU_VERSION_SPEC_CU13}"
    return f"{GPU_PACKAGE}{GPU_VERSION_SPEC_CU12}"


# ---------------------------------------------------------------------------
# dist-info 探针（不 import .pyd）
# ---------------------------------------------------------------------------


def _query_dist_info() -> tuple[Optional[str], Optional[str]]:
    """从 dist-info 读三个互斥包的安装状态。返回 (pkg_name, version)。"""
    try:
        from importlib.metadata import PackageNotFoundError, version as _ver
        for pkg in _MUTUALLY_EXCLUSIVE_PACKAGES:
            try:
                return pkg, _ver(pkg)
            except PackageNotFoundError:
                continue
    except Exception:  # noqa: BLE001
        pass
    return None, None


# ---------------------------------------------------------------------------
# CUDA 共享库预加载（PP9.5）
# ---------------------------------------------------------------------------


_PRELOAD_RESULT: Optional[dict[str, Any]] = None
_CUDA_LOAD_ERROR: Optional[str] = None


def _has_system_cuda_libs() -> bool:
    """Linux 系统是否自带**完整** CUDA 运行时（cuBLAS + cuDNN 都在 ld 路径）。

    有**完整**系统 CUDA 时跳过 PP9.5 preload —— torch wheel 自带的 CUDA so
    （cu128 → cuBLAS 12.8）与 onnxruntime-gpu wheel 编译目标的 CUDA so
    （typically 12.x 某子版本）ABI 不匹配；RTLD_GLOBAL 把 torch 的强行塞进
    全局符号表后，onnxruntime 后续 dlopen cuBLAS 解到错位版本 → 推理时
    CUBLAS_STATUS_INVALID_VALUE。系统 CUDA 完整时让 onnxruntime 直接 dlopen
    系统版本反而是对的。

    **关键**：必须 cuBLAS + cuDNN 都在系统里才算完整。云镜像装 CUDA Toolkit
    （带 cuBLAS）但**没装** cuDNN 极常见（cuDNN 要 NVIDIA Developer 账号单独
    下）；只检测 cuBLAS 会误判 → preload 跳过 → onnxruntime dlopen
    libcudnn.so.9 失败 → 静默降 CPU。这种「部分系统 CUDA」场景必须让 torch
    wheel preload 兜底补 cuDNN（torch GPU build 自带 cuDNN 9.x 在
    nvidia.cudnn 子包里）。

    检测分两步，都满足才返回 True：
    1. CUDA Toolkit：CUDA_HOME / CUDA_PATH 指向带 lib64 / 默认 /usr/local/cuda
       存在 / ld 路径里有 libcublas —— 任一命中
    2. cuDNN：ld 路径里有 libcudnn —— 必须命中
    """
    import ctypes.util  # noqa: PLC0415  仅 Linux 路径用，避免顶层 import 副作用
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or ""
    has_toolkit = False
    if cuda_home and os.path.isdir(os.path.join(cuda_home, "lib64")):
        has_toolkit = True
    elif os.path.isdir("/usr/local/cuda/lib64"):
        has_toolkit = True
    elif ctypes.util.find_library("cublas"):
        has_toolkit = True
    if not has_toolkit:
        return False
    # 关键：cuDNN 同样必须在系统 ld 路径里。云镜像装 CUDA Toolkit（含 cuBLAS）
    # 但没装 cuDNN 是非常常见的场景；只检测 cuBLAS 会误判 → preload 被跳过 →
    # onnxruntime dlopen libcudnn.so.9 失败 → 静默降 CPU。这种部分系统 CUDA
    # 场景下让 torch wheel preload 兜底补 cuDNN（torch GPU build 在
    # nvidia.cudnn 子包里自带 cuDNN 9.x）。
    return bool(ctypes.util.find_library("cudnn"))


def _resolve_cuda_major() -> Optional[int]:
    """决定 onnxruntime-gpu 该走哪个 CUDA 大版本（12 / 13 / None）。

    锚定在 **torch 实际编进去的 CUDA** 上：预加载靠 torch 自带的 nvidia wheel，
    ORT build 也必须匹配同一个 major，否则同进程 cu12/cu13 混用 → 推理期 ABI 错位
    （CUBLAS_STATUS_*）或 import 期 dlopen 挂（如 libcudart.so.13 not found —— 即
    本修复要解决的回归：旧 `>=1.20` 拉到 ORT 1.27 CUDA 13，而项目 torch 是 cu12）。

    解析顺序：
    1. `torch.version.cuda`（形如 "12.8" → 12、"13.0" → 13）—— torch CUDA build 的
       CUDA 大版本，最权威。
    2. torch 是 CPU build / 没装 → 退到 nvidia-smi 驱动版本，复用 torch.recommend_cu_tag
       （函数内 import 避开 torch.py ↔ onnxruntime.py 的循环 import）拿 cu tag：
       `cu128`→12、`cu130`→13。当前 recommend_cu_tag 表最高 cu128，故这条 fallback
       今天恒返回 12。
    3. 都拿不到（无 GPU、无 torch）→ None；调用方按 DEFAULT_CUDA_MAJOR（=12）。

    DCU 上恒为 None：那里没有「CUDA 大版本」这个概念，而 fallback 那条会拿 hy-smi
    的驱动版本（形如 "1.4.1"）喂给 recommend_cu_tag，硬凑出一个毫无意义的 cu 标签
    显示在 UI 上。DCU 走不到任何真正用这个值的路径（gpu 目标被拒、CUDA wheel 跳过），
    所以直接短路返回 None，让 UI 隐藏这一行。
    """
    if _backend() == "dcu":
        return None
    try:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415
        cuda_v = getattr(torch.version, "cuda", None)
        if cuda_v:
            return int(str(cuda_v).split(".")[0])
    except (ImportError, ValueError, TypeError):
        pass
    try:
        from .torch import recommend_cu_tag  # noqa: PLC0415  函数内 import 断循环
    except ImportError:
        return None
    tag = recommend_cu_tag(detect_cuda().get("driver_version"))
    # tag 形如 "cu128" → 128 // 10 = 12；"cu118" → 11；"cu130" → 13；"cpu" → None
    if tag.startswith("cu") and tag[2:].isdigit():
        return int(tag[2:]) // 10
    return None


def _add_torch_dll_dirs_windows() -> dict[str, Any]:
    """PR — Windows 上把 torch 自带的 CUDA DLL 目录加入 Python DLL 搜索路径。

    Python 3.8+ Windows 出于安全废除了 PATH 自动 dlopen native DLL —— 必须
    `os.add_dll_directory()` 主动声明。onnxruntime 在 import 期 dlopen
    `cublasLt64_12.dll` / `cudnn_*.dll` 时找不到，`get_available_providers()`
    照样列 CUDAExecutionProvider，但 InferenceSession 内部 silently 降 CPU
    （onnx_tagger_base._create_session 已经能识别这种降级）。

    torch GPU build wheel 把全套 CUDA DLL 放在 `site-packages/torch/lib/`
    （cublasLt64_12 / cudnn*_9 / curand64_10 / cufft64_11 / cusparse64_12 /
    cudart64_12 / nvrtc）。加进 DLL search path，后续 onnxruntime 的 dlopen
    就能找到。

    返回 `{"added", "errors", "candidates"}`：
    - `added`：成功 add_dll_directory 的目录列表
    - `errors`：尝试但失败的 (dir, reason) 列表
    - `candidates`：发现的候选目录数（=0 表示 venv 没装 torch GPU build）
    """
    added: list[str] = []
    errors: list[tuple[str, str]] = []
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return {"added": added, "errors": errors, "candidates": 0}
    lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if not os.path.isdir(lib):
        return {"added": added, "errors": errors, "candidates": 0}
    try:
        # Python 3.8+ Windows API；其他平台没有
        os.add_dll_directory(lib)  # type: ignore[attr-defined]
        added.append(lib)
    except (OSError, AttributeError) as exc:
        errors.append((lib, str(exc)))
    return {"added": added, "errors": errors, "candidates": 1}


def _preload_torch_cuda_libs() -> dict[str, Any]:
    """跨平台预加载 torch 自带的 CUDA 库，让 onnxruntime-gpu dlopen 找得到。

    背景：onnxruntime-gpu wheel 不打包 CUDA runtime；用户机器没系统装 CUDA
    时，CUDA EP 在 `get_available_providers()` 里看着可用，但创 session 时
    dlopen 失败（Linux: `libcurand.so.10`；Windows: `cublasLt64_12.dll`）。
    onnxruntime 不抛异常，会**静默降级到 CPU**（onnx_tagger_base 已有检测）。

    PyTorch GPU build 自带所有需要的 CUDA 库：
    - **Linux**：装到 `site-packages/nvidia/*/lib/` —— `ctypes.CDLL(RTLD_GLOBAL)`
      预加载到全局符号表
    - **Windows**：装到 `site-packages/torch/lib/` —— `os.add_dll_directory()`
      加入 DLL 搜索路径（Python 3.8+ 必需）

    **注意**：Linux 上系统已有 CUDA（如 nvidia docker 镜像）时跳过 preload。
    torch wheel 的 CUDA 版本（cu128 = cuBLAS 12.8）与 onnxruntime-gpu 编译目标
    的 CUDA 子版本不一致时，强行 RTLD_GLOBAL 覆盖会导致推理时
    CUBLAS_STATUS_INVALID_VALUE。Windows 上 `add_dll_directory` 只是把目录加进
    搜索路径，不强行覆盖已加载的符号，所以无此问题。

    只对**当前进程**生效；server 子进程必须自己再跑一次（本模块在 import 时
    自动跑）。

    返回 `{"applied", "platform_skip", "system_cuda_skip", "backend_skip",
           "preloaded", "errors", "candidates"}`：
    - `backend_skip=True`：非 NVIDIA 后端（海光 DCU），整体跳过 —— 见下方说明
    - `platform_skip=True`：非 Linux / 非 Windows（如 macOS），整体跳过
    - `system_cuda_skip=True`：Linux 系统 CUDA 路径，跳过 preload 让 onnxruntime
      自己 dlopen 系统提供的版本
    - `preloaded`：成功 dlopen / add_dll_directory 的绝对路径列表
    - `errors`：尝试但失败的 (path, reason) 列表
    - `candidates`：检视的候选数（Linux: nvidia.* 子包；Windows: torch/lib 目录）
    """
    # DCU 上整段跳过，且必须放在**平台判断之前**（DCU 也是 Linux，会落进下面那条
    # 真正 dlopen 的分支）。两个理由：
    # 1. `site-packages/nvidia/*` 是 NVIDIA torch wheel 的依赖包，DTK 镜像里根本
    #    不存在 —— 走进去只是 8 次 ImportError 空转。
    # 2. DTK 配套的 onnxruntime 自己链 DTK 运行时（libamdhip64 等，由 DTK 装在系统
    #    ld 路径里），不存在 PP9.5 要解决的「wheel 不带 runtime so」问题；真硬塞
    #    NVIDIA so 进全局符号表反而是引入风险。
    if _backend() == "dcu":
        return {
            "applied": False,
            "platform_skip": False,
            "system_cuda_skip": False,
            "backend_skip": True,
            "preloaded": [],
            "errors": [],
            "candidates": 0,
        }
    if sys.platform == "win32":
        wres = _add_torch_dll_dirs_windows()
        return {
            "applied": True,
            "platform_skip": False,
            "system_cuda_skip": False,
            "backend_skip": False,
            "preloaded": wres["added"],
            "errors": wres["errors"],
            "candidates": wres["candidates"],
        }
    if not sys.platform.startswith("linux"):
        return {
            "applied": False,
            "platform_skip": True,
            "system_cuda_skip": False,
            "backend_skip": False,
            "preloaded": [],
            "errors": [],
            "candidates": 0,
        }
    if _has_system_cuda_libs():
        return {
            "applied": False,
            "platform_skip": False,
            "system_cuda_skip": True,
            "backend_skip": False,
            "preloaded": [],
            "errors": [],
            "candidates": 0,
        }
    preloaded: list[str] = []
    errors: list[tuple[str, str]] = []
    seen: set[str] = set()
    candidates = 0
    for pkg in _TORCH_NVIDIA_LIB_PKGS_LINUX:
        try:
            mod = importlib.import_module(pkg)
        except ImportError:
            continue
        candidates += 1
        for base in getattr(mod, "__path__", []):
            lib_dir = os.path.join(base, "lib")
            if not os.path.isdir(lib_dir):
                continue
            # glob 该子包自带的全部 lib*.so*：cu12 是 .so.12 / libcudnn.so.9，
            # cu13 是 .so.13 / libcudnn.so.10，glob 自动适配，无需维护 soname 表。
            for so in sorted(os.listdir(lib_dir)):
                if not so.startswith("lib") or ".so" not in so:
                    continue
                candidate = os.path.join(lib_dir, so)
                if candidate in seen:
                    continue
                try:
                    ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                except OSError as exc:
                    errors.append((candidate, str(exc)))
                    continue
                preloaded.append(candidate)
                seen.add(candidate)
    return {
        "applied": True,
        "platform_skip": False,
        "system_cuda_skip": False,
        "backend_skip": False,
        "preloaded": preloaded,
        "errors": errors,
        "candidates": candidates,
    }


def _ensure_preload() -> dict[str, Any]:
    """幂等触发预加载；首次调用时跑一次，后续返回 cached 结果。

    onnxruntime 未装时整体 skip —— preload 唯一作用是给后续 import onnxruntime
    的 CUDA EP dlopen 兜底；没装就无意义。装上后必须重启 Studio（C extension
    不能热替换），新进程 import 本模块时会再次触发，依然正确生效。
    """
    global _PRELOAD_RESULT
    if _PRELOAD_RESULT is not None:
        return _PRELOAD_RESULT
    if _query_dist_info()[0] is None:
        _PRELOAD_RESULT = {
            "applied": False,
            "platform_skip": False,
            "system_cuda_skip": False,
            "backend_skip": False,
            "not_installed_skip": True,
            "preloaded": [],
            "errors": [],
            "candidates": 0,
        }
        return _PRELOAD_RESULT
    _PRELOAD_RESULT = _preload_torch_cuda_libs()
    if _PRELOAD_RESULT.get("backend_skip"):
        logger.info(
            "[onnx_setup] 后端为 %s，跳过 NVIDIA CUDA 库预加载"
            "（DTK 配套 onnxruntime 自带运行时依赖）",
            _vendor_label(),
        )
    elif sys.platform == "win32" and _PRELOAD_RESULT["preloaded"]:
        logger.info(
            "[onnx_setup] DLL 搜索路径已加入 torch/lib（onnxruntime-gpu CUDA dlopen 用）"
        )
    elif _PRELOAD_RESULT["preloaded"]:
        logger.info(
            "[onnx_setup] 预加载 torch 自带 CUDA 库 %d 个: %s",
            len(_PRELOAD_RESULT["preloaded"]),
            ", ".join(
                os.path.basename(p) for p in _PRELOAD_RESULT["preloaded"]
            ),
        )
    elif _PRELOAD_RESULT.get("system_cuda_skip"):
        logger.info(
            "[onnx_setup] 检测到系统 CUDA，跳过 torch wheel preload（避免 cuBLAS 版本错位）"
        )
    elif _PRELOAD_RESULT["applied"] and _PRELOAD_RESULT["candidates"] == 0:
        logger.debug(
            "[onnx_setup] 未发现 torch 自带 CUDA wheel；GPU EP 依赖系统 CUDA"
        )
    return _PRELOAD_RESULT


def record_cuda_load_error(msg: Optional[str]) -> None:
    """wd14_tagger.prepare 创 InferenceSession 失败 → 调本函数 stash 原因。

    None 表示成功 / 清空（成功的 session 创建会清旧错误）。
    """
    global _CUDA_LOAD_ERROR
    _CUDA_LOAD_ERROR = msg


def get_cuda_load_error() -> Optional[str]:
    return _CUDA_LOAD_ERROR


# 模块加载即触发预加载 —— 必须在任何地方 `import onnxruntime` 之前生效。
# server.py 顶层 `from .services import onnxruntime_setup` 已经覆盖 server 子进程；
# cli.py 也在 cmd_run 早期 import 本模块。
_ensure_preload()


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def detect_cuda() -> dict[str, Any]:
    """GPU 硬件探针。返回 ``{"available", "driver_version", "gpu_name", "backend"}``。

    nvidia-smi 不需要 root，是最低成本的 GPU 检测；找不到 / 跑失败都视作无 GPU。

    **函数名与前三个 key 刻意不改**：``torch.py`` / ``cli.py`` / ``tools/bench_wd14.py``
    都在消费它，改名等于同时改三个不属于本次范围的文件。语义从「nvidia-smi 探针」
    放宽成「当前后端的 GPU 探针」——DCU 上走 ``accelerator.probe_stdlib()``（hy-smi /
    rocm-smi / ``/dev/kfd``），这样 ``cli.py`` 的启动期 `has_gpu` 判断与
    ``_decide_target("auto")`` 在 DCU 上也能得到正确答案，而不是「没 GPU」。

    ``backend`` 是**新增**字段（加法，老消费方不受影响），值为 ``cuda`` / ``dcu``
    / ``cpu``，让 UI 能把「NVIDIA 驱动」那行标签换成正确的厂商名。
    """
    if _backend() == "dcu":
        # DCU：nvidia-smi 不存在，probe_stdlib() 会依次试 hy-smi / rocm-smi /
        # /dev/kfd。它每次真跑 smi（不缓存），调用频率是页面刷新级，可接受。
        try:
            probe = accelerator.probe_stdlib()
        except Exception as exc:  # noqa: BLE001  探测失败不该让 status endpoint 500
            logger.debug("DCU 硬件探测失败: %s", exc)
            return {
                "available": False, "driver_version": None,
                "gpu_name": None, "backend": "dcu",
            }
        # backend=="dcu" 已由 torch 侧确认（_backend() 读的是 torch.version.hip），
        # 所以这里 available 只表示「smi / 设备节点也能确认」；probe 退化成 cpu
        # 说明容器没挂 /dev/kfd 或 DTK 装得不全，报 False 让 UI 提示排查。
        return {
            "available": probe.backend == "dcu",
            "driver_version": probe.driver_version,
            "gpu_name": probe.gpu_name,
            "backend": "dcu",
        }

    # 注意 backend 与 available 是**两个独立事实**，不要互推：backend 来自 torch
    # build（cpu 版 torch → "cpu"），available 来自硬件探针。两者不一致正是
    # torch.py 的 `is_cpu_with_gpu` 误装诊断要抓的场景。
    bk = _backend()
    nv = shutil.which("nvidia-smi")
    if not nv:
        return {"available": False, "driver_version": None, "gpu_name": None, "backend": bk}
    try:
        out = subprocess.run(
            [
                nv,
                "--query-gpu=driver_version,name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("nvidia-smi exec failed: %s", exc)
        return {"available": False, "driver_version": None, "gpu_name": None, "backend": bk}
    if out.returncode != 0:
        return {"available": False, "driver_version": None, "gpu_name": None, "backend": bk}
    line = (out.stdout or "").strip().splitlines()
    if not line:
        return {"available": False, "driver_version": None, "gpu_name": None, "backend": bk}
    parts = [p.strip() for p in line[0].split(",", 1)]
    driver = parts[0] if parts else None
    name = parts[1] if len(parts) > 1 else None
    return {"available": True, "driver_version": driver, "gpu_name": name, "backend": bk}


def current_runtime() -> dict[str, Any]:
    """返回当前进程视角的 onnxruntime 信息。

    `installed` 来自 dist-info（pip 视角）；`providers` 是 import 后实际可用 EP（已加载
    的 native 模块视角）。两者**可能不一致** —— 装完包不重启 → dist-info 显示新包，
    providers 仍是旧包的。`restart_required` 表示这种状态。
    """
    installed_pkg, installed_ver = _query_dist_info()
    process_version: Optional[str] = None
    providers: list[str] = []
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
        providers = list(ort.get_available_providers())
        process_version = getattr(ort, "__version__", None)
    except ImportError:
        pass

    # 检测「pip 装的包」与「进程里已 import 的 native 模块」不一致 —— onnxruntime
    # 是 C extension，pip 卸装重装不会热替换已 import 的 .pyd，必须重启才换 EP。
    #
    # 当前后端的 GPU EP 名：DCU→MIGraphXExecutionProvider，其余→CUDAExecutionProvider。
    # 下面的 cuda_available 与「装了 CPU 包却还有加速 EP」判定都锚定它，不再硬编码。
    gpu_provider = _gpu_ep_name()

    # 判定只看「装的包类型 ↔ 进程里实际加载的 EP」，**不比版本号字符串**：
    # onnxruntime-directml 的 dist 版本（如 1.24.4）与它内部捆绑的 onnxruntime 核心
    # 版本（ort.__version__，如 1.27.0）是两条独立版本线、天然不相等；比版本号会让
    # DirectML 用户永久误报「需重启」，重启多少次都消不掉。EP 一致性才是可靠信号，
    # 也正是本功能的目的（让 EP 切换生效）。
    #
    # 加速 EP 集合按后端拼：DCU 上把 MIGraphX 也算进去，这样「从 DTK 配套 onnxruntime
    # 换成 PyPI CPU 包但没重启」同样能被识别成 restart_required。CUDA EP 保留在集合里
    # （DCU 上它永远不会出现，留着无副作用，且 NVIDIA 路径逐字节不变）。
    _ACCEL_EPS = tuple(dict.fromkeys(
        ["CUDAExecutionProvider", "DmlExecutionProvider", gpu_provider]
    ))
    restart_required = False
    if installed_pkg is not None and process_version is not None:
        if installed_pkg == GPU_PACKAGE and "CUDAExecutionProvider" not in providers:
            # 装了 GPU 包但进程没 CUDA EP → 仍在跑旧（CPU / DirectML）包
            restart_required = True
        elif installed_pkg == DIRECTML_PACKAGE and "DmlExecutionProvider" not in providers:
            # 装了 DirectML 包但进程没 Dml EP → 仍在跑旧（CPU / GPU）包
            restart_required = True
        elif installed_pkg == CPU_PACKAGE and any(ep in providers for ep in _ACCEL_EPS):
            # 装了 CPU 包但进程还有加速 EP → 仍在跑旧（GPU / DirectML）包
            restart_required = True

    # torch 的 CUDA 大版本（onnxruntime-gpu 选 build 的锚点）；UI / diagnose 自查用。
    torch_cuda_major = _resolve_cuda_major()
    # 错位启发式：cuda_load_error 里含 .so.13 但 torch 是 12（或反之）→ 装出来的
    # ORT build 与 torch 不同 major（正是本修复要避免的回归）。best-effort，仅提示。
    ort_cuda_major_mismatch = False
    load_err = _CUDA_LOAD_ERROR or ""
    if load_err and torch_cuda_major is not None:
        if ".so.13" in load_err and torch_cuda_major != 13:
            ort_cuda_major_mismatch = True
        elif ".so.12" in load_err and torch_cuda_major != 12:
            ort_cuda_major_mismatch = True

    return {
        "installed": installed_pkg,
        "version": installed_ver or process_version,
        "providers": providers,
        # 语义已放宽成「**当前后端的** GPU EP 可用」（DCU 上 = MIGraphX EP）。key 名
        # 保持 cuda_available 不改：前端 / cli.py / tools/diagnose_onnx_gpu.py 都在读它，
        # 改名的收益（名字更准）远小于同步改动的代价。实际 EP 名见 gpu_provider。
        "cuda_available": gpu_provider in providers,
        "directml_available": "DmlExecutionProvider" in providers,
        # 当前后端期望的 GPU EP 名（DCU=MIGraphX，其余=CUDA）。UI 用它把「CUDA」
        # 字样换成实际 EP 名，避免在 DCU 上显示 CUDA 误导用户。
        "gpu_provider": gpu_provider,
        # 后端标识 + 面向用户的厂商名（加法）。前端据此置灰不适用的装包按钮。
        "backend": _backend(),
        "vendor_label": _vendor_label(),
        # 平台标识：前端按平台 disable DirectML/GPU 按钮（DirectML 仅 Windows；
        # CUDA runtime wheel 仅 Linux 有；CPU 全平台可用）
        "platform": sys.platform,
        "restart_required": restart_required,
        # PP9.5 — 创 InferenceSession 时实际 dlopen 报的错（如 `libcurand.so.10`
        # 缺失）；wd14_tagger.prepare 降 CPU 后填进来。None=没碰过 / 上次成功。
        "cuda_load_error": _CUDA_LOAD_ERROR,
        # PP9.5 — torch 自带 CUDA so 预加载结果（Linux only）；UI 诊断用
        "preload": _PRELOAD_RESULT,
        # torch 的 CUDA 大版本（onnxruntime-gpu build 锚点）+ 是否与已装 ORT 错位
        "torch_cuda_major": torch_cuda_major,
        "ort_cuda_major_mismatch": ort_cuda_major_mismatch,
    }


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


_PIP_FALLBACK_MIRROR = "https://mirrors.cloud.tencent.com/pypi/simple/"


def _pip(args: list[str], *, mirror: str = "") -> tuple[int, str]:
    """跑 `<sys.executable> -m pip <args>`；返回 (rc, combined_output)。

    mirror 非空时追加 `-i {mirror}`（用于镜像 fallback 重试）。
    """
    cmd = [sys.executable, "-m", "pip", *args]
    if mirror:
        cmd += ["-i", mirror]
    logger.info("[onnx_setup] %s", " ".join(cmd))
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # pip install 几分钟级别
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return 1, f"pip 超时（10 分钟）: {exc}"
    except Exception as exc:  # noqa: BLE001
        return 1, f"pip 调用失败: {exc}"
    text = (out.stdout or "") + (out.stderr or "")
    return out.returncode, text


def _dcu_migraphx_active() -> bool:
    """DCU 上当前进程是否已有可用的 MIGraphX EP（= DTK 配套 onnxruntime 装好了）。

    是「装包会不会造成破坏」的判据，不是能力查询：为 True 时任何 pip 装包都会先
    ``uninstall onnxruntime``（三个互斥包同名），把镜像里那个自带 MIGraphX EP 的
    DTK build 卸掉，再从 PyPI 装回一个纯 CPU build —— GPU 打标能力就这么没了，
    而且 pip 装不回来（DTK wheel 不在 PyPI 上）。
    """
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]  # noqa: PLC0415
        return "MIGraphXExecutionProvider" in ort.get_available_providers()
    except Exception:  # noqa: BLE001  未装 / import 失败都算「没有可保护的东西」
        return False


def _decide_target(target: str) -> str:
    """auto/gpu/cpu/directml → 实际包名（带版本约束）。

    GPU 路径的版本约束按 _resolve_cuda_major() 分流 cu12/cu13，保证 ORT build 与
    torch 同 major（否则同进程 cu12/cu13 混用 ABI 错位）。

    auto 路径按平台分流：
    - Windows + GPU → DirectML（绕开 CUDA dlopen 问题，跨厂商）
    - Linux + GPU → onnxruntime-gpu（native CUDA EP 最优），版本约束按 torch major
    - 无 GPU → CPU 包

    **DCU 上 gpu 目标抛 RuntimeError、auto 落到 CPU 包** —— 决策理由见
    :func:`_decide_target_dcu`。
    """
    if _backend() == "dcu":
        return _decide_target_dcu(target)
    if target == "gpu":
        return _gpu_version_spec_for(_resolve_cuda_major())
    if target == "cpu":
        return f"{CPU_PACKAGE}{CPU_VERSION_SPEC}"
    if target == "directml":
        return f"{DIRECTML_PACKAGE}{DIRECTML_VERSION_SPEC}"
    if target == "auto":
        cuda = detect_cuda()
        if cuda["available"]:
            if sys.platform == "win32":
                return f"{DIRECTML_PACKAGE}{DIRECTML_VERSION_SPEC}"
            return _gpu_version_spec_for(_resolve_cuda_major())
        return f"{CPU_PACKAGE}{CPU_VERSION_SPEC}"
    raise ValueError(f"非法 target: {target!r}（应为 auto/gpu/cpu/directml）")


def _decide_target_dcu(target: str) -> str:
    """DCU 上的目标决策：``gpu`` 拒绝、``auto`` / ``cpu`` 给 CPU 包、``directml`` 拒绝。

    **为什么 gpu 拒绝而不是静默降 CPU**：``onnxruntime-gpu`` 是 CUDA build，装到
    DCU 上 import 期就 dlopen 挂 libcudart，而且它与 CPU 包**同名互斥** —— 装它等于
    先把能用的包卸掉，再换一个 import 不起来的，打标功能直接归零（比降 CPU 更糟）。
    静默降 CPU 也不行：用户明确点了「装 GPU 版」，给他一个 CPU 包却报成功，他会一直
    以为打标在跑 GPU、然后困惑于为什么这么慢。抛错 + 说清「GPU 打标要从 DTK 渠道装
    配套 onnxruntime」才是能让他真正解决问题的信息。

    **为什么 auto 给 CPU 包而不是拒绝**：``auto`` 的语义是「你别管，给我一个能用的」，
    它同时是 UI 上的主按钮和 ``cli.py`` 启动期缺包时的建议路径 —— 必须**永远能产出一个
    可用状态**。DCU 上唯一能靠 pip 拿到的可用包就是 CPU 版；装上后打标能跑（慢但可用），
    这比「点了主按钮报错、功能完全不可用」好。上层 UI 同时显示「GPU 打标需从 DTK 渠道
    装配套 onnxruntime」，用户想要 GPU 有明确的下一步。

    ``directml`` 同样拒绝：DirectML 是 Windows DX12 后端，DCU 只有 Linux。
    """
    if target == "cpu":
        return f"{CPU_PACKAGE}{CPU_VERSION_SPEC}"
    if target == "auto":
        return f"{CPU_PACKAGE}{CPU_VERSION_SPEC}"
    if target == "gpu":
        raise RuntimeError(
            f"当前后端是 {_vendor_label()}，不能安装 onnxruntime-gpu。\n"
            "onnxruntime-gpu 是 NVIDIA CUDA build（链接 libcudart / libcudnn），"
            "在 DCU 上 import 就会失败；而且它与 CPU 版 onnxruntime 同名互斥，"
            "装它会先把当前能用的包卸掉 —— 结果是打标功能完全不可用。\n"
            "DCU 上要用 GPU 打标，需从 DTK 渠道安装海光配套的 onnxruntime"
            "（自带 MIGraphXExecutionProvider，PyPI 上没有这个 build）：\n"
            "  1. 从光合开发者社区 / DTK 配套仓库取与镜像 DTK 版本匹配的 onnxruntime 包\n"
            "  2. pip install 该包（装完重启 Studio，C extension 不能热替换）\n"
            "  3. 本页会自动识别 MIGraphX EP 并显示为 GPU 可用\n"
            "只想先把打标跑起来的话，点「自动检测」装 CPU 版即可（慢但可用）。"
        )
    if target == "directml":
        raise RuntimeError(
            f"当前后端是 {_vendor_label()}，不能安装 onnxruntime-directml。"
            "DirectML 是 Windows 上的 DX12 后端，DCU 只有 Linux 环境。"
        )
    raise ValueError(f"非法 target: {target!r}（应为 auto/gpu/cpu/directml）")


def _is_dist_installed(pkg: str) -> bool:
    """dist-info 里有没有这个包；不 import，避免触发 native 模块加载。"""
    try:
        from importlib.metadata import PackageNotFoundError, version as _ver
        try:
            _ver(pkg)
            return True
        except PackageNotFoundError:
            return False
    except Exception:  # noqa: BLE001
        return False


def _install_cuda_runtime_wheels(major: Optional[int] = None) -> dict[str, Any]:
    """PP9.6 — Linux 上把 onnxruntime-gpu 跑起来需要的 CUDA runtime wheels 装上。

    `major`：CUDA 大版本（12/13）；None 时 _resolve_cuda_major() 推断（拿不到走
    DEFAULT_CUDA_MAJOR）。按 major 选 cu12（nvidia-*-cu12）或 cu13（去后缀名）两套。
    与 _decide_target 选的 ORT 版本约束用同一个 major（都锚定 torch.version.cuda）。

    返回 `{"installed": [...新装的], "skipped": [...原本就有的], "platform_skip": bool,
           "cuda_major": int|None, "stdout": str}`。失败抛 RuntimeError，并在抛之前
    **回滚本次刚装的包**（保持 venv 不被污染）。

    cuDNN 单独处理：原本就有就不动（避免撞 torch 锁的版本）；没有才补。

    DCU 上整段跳过（``backend_skip=True``）：这些 ``nvidia-*`` wheel 与 DCU 毫无关系，
    装进去只是白占几个 GB 磁盘 + 污染 venv。实际走不到这里（DCU 的 gpu 目标已被
    _decide_target_dcu 拒掉、CPU 路径本就不调本函数），保留这道判断是防御性的
    —— 万一将来有别的调用点，不该在 DCU 上偷偷装 NVIDIA 包。
    """
    if _backend() == "dcu":
        return {
            "installed": [],
            "skipped": [],
            "platform_skip": False,
            "backend_skip": True,
            "cuda_major": None,
            "stdout": "non-nvidia backend; skip nvidia cuda runtime wheels",
        }
    if not sys.platform.startswith("linux"):
        # Windows / macOS：nvidia CUDA runtime wheel 不可用；用户应靠系统 CUDA Toolkit
        return {
            "installed": [],
            "skipped": [],
            "platform_skip": True,
            "backend_skip": False,
            "cuda_major": major,
            "stdout": "non-linux platform; skip nvidia cuda runtime wheels",
        }
    if major is None:
        major = _resolve_cuda_major() or DEFAULT_CUDA_MAJOR
    runtime_wheels, cudnn_wheel = _cuda_wheels_for(major)
    targets: list[str] = []
    skipped: list[str] = []
    # cuDNN：只在缺时装（torch GPU build 通常已带）
    if _is_dist_installed(cudnn_wheel):
        skipped.append(cudnn_wheel)
    else:
        targets.append(cudnn_wheel)
    # 其余 6 个：缺啥装啥
    for pkg in runtime_wheels:
        if _is_dist_installed(pkg):
            skipped.append(pkg)
        else:
            targets.append(pkg)
    if not targets:
        return {
            "installed": [],
            "skipped": skipped,
            "platform_skip": False,
            "backend_skip": False,
            "cuda_major": major,
            "stdout": "all CUDA runtime wheels already present",
        }
    rc, out = _pip(["install", *targets])
    if rc != 0:
        logger.warning("[onnx_setup] CUDA wheels pip 官方源失败，切换腾讯镜像重试...")
        rc, out = _pip(["install", *targets], mirror=_PIP_FALLBACK_MIRROR)
    if rc != 0:
        # 回滚：把本次想装的从 venv 里再卸掉，保持装包前的状态
        # （pip install 失败时部分包可能已装；不区分，统一卸）
        rb_rc, rb_out = _pip(["uninstall", "-y", *targets])
        raise RuntimeError(
            f"安装 CUDA runtime wheels 失败（rc={rc}）:\n{out}\n"
            f"--- rollback (rc={rb_rc}) ---\n{rb_out}"
        )
    return {
        "installed": targets,
        "skipped": skipped,
        "platform_skip": False,
        "backend_skip": False,
        "cuda_major": major,
        "stdout": out,
    }


def install_runtime(target: str = "auto") -> dict[str, Any]:
    """先 uninstall 三个互斥包再装目标。

    target: "auto" | "gpu" | "cpu" | "directml"
    返回 `{"target", "installed_pkg", "installed_version", "restart_required": True,
           "stdout", "cuda_runtime"}`，最后一个字段是 PP9.6 装的 nvidia CUDA runtime
    wheels 报告（按 torch 的 CUDA major 选 cu12 / cu13 两套；仅 GPU 路径；CPU /
    DirectML 路径为 None）。失败抛 RuntimeError。

    **重要**：onnxruntime 是 C extension，pip 卸装重装后**当前进程**里已 import 的
    .pyd/.so 不会被热替换 —— 必须重启 Studio 才能切换 EP。所以本函数不再尝试 reload；
    返回 `restart_required=True` 让 UI 提示用户重启。

    DCU 上：``gpu`` / ``directml`` 目标被 _decide_target_dcu 拒掉（抛 RuntimeError）；
    已装 DTK 配套 onnxruntime 时**任何**装包请求都被拒（见下方护栏）。
    """
    # 护栏：DCU 上已经有可用的 MIGraphX EP 时，绝不能走 pip —— 第一步 uninstall 就会
    # 把镜像/DTK 装的那个包卸掉（三个互斥包与它同名），然后从 PyPI 换回一个纯 CPU
    # build，GPU 打标能力永久丢失且 pip 装不回来（DTK wheel 不在 PyPI 上），只能重建
    # 容器。这与 accelerator.should_manage_torch_install() 保护 DTK torch 是同一类问题。
    if _backend() == "dcu" and _dcu_migraphx_active():
        raise RuntimeError(
            f"当前已装 DTK 配套的 onnxruntime（MIGraphXExecutionProvider 可用），"
            f"{_vendor_label()} 上不执行任何 pip 装包。\n"
            "onnxruntime / onnxruntime-gpu / onnxruntime-directml 三个包与它**同名互斥**，"
            "装包第一步的 pip uninstall 会把它卸掉，之后只能从 PyPI 装回纯 CPU build"
            "（DTK wheel 不在 PyPI 上，pip 装不回来，只能重建容器）。\n"
            "当前状态就是 DCU 上的最佳状态，不需要任何操作。"
        )

    spec = _decide_target(target)
    rc1, log1 = _pip(["uninstall", "-y", *_MUTUALLY_EXCLUSIVE_PACKAGES])
    rc2, log2 = _pip(["install", "--upgrade", spec])
    if rc2 != 0:
        logger.warning("[onnx_setup] pip 官方源失败，切换腾讯镜像重试...")
        rc2, log2 = _pip(["install", "--upgrade", spec], mirror=_PIP_FALLBACK_MIRROR)
    if rc2 != 0:
        raise RuntimeError(f"安装 {spec} 失败（rc={rc2}）:\n{log2}")

    # PP9.6 — GPU 路径补齐 CUDA runtime wheels（onnxruntime-gpu 不打包它们）。
    # CPU 路径或 auto 检测为 CPU 时跳过。
    cuda_runtime: Optional[dict[str, Any]] = None
    if GPU_PACKAGE in spec:
        try:
            cuda_runtime = _install_cuda_runtime_wheels()
        except RuntimeError as exc:
            # CUDA wheels 装失败不致命：onnxruntime-gpu 已装上，让用户去 Settings 页
            # 看到 cuda_load_error + 手动修。日志记下原因，UI 也能拿到。
            logger.error("[onnx_setup] CUDA runtime wheels 装失败: %s", exc)
            cuda_runtime = {
                "installed": [],
                "skipped": [],
                "platform_skip": False,
                "cuda_major": None,
                "stdout": str(exc),
                "error": str(exc),
            }

    # 直接读 dist-info 拿新装的版本（不 import；进程里仍是旧的 native 模块）
    new_pkg, new_ver = _query_dist_info()
    return {
        "target": spec,
        "installed_pkg": new_pkg,
        "installed_version": new_ver,
        "restart_required": True,
        "stdout": log1 + log2,
        "cuda_runtime": cuda_runtime,
    }
