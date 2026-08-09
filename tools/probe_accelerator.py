#!/usr/bin/env python
"""加速器环境探测：把移植 / 排错需要的事实一次性打出来。

**用途**：在陌生的 GPU 环境（海光 DCU + DTK 镜像、NVIDIA 容器、云租赁机）里跑一遍，
确认这套训练器依赖的每个能力到底可不可用 —— 而不是靠猜。典型场景是新后端移植前
的基线采集，以及用户报「训练崩了」时让对方跑一遍贴结果。

**怎么跑**（在项目根目录）：

    python tools/probe_accelerator.py              # 人类可读 + 末尾 JSON
    python tools/probe_accelerator.py --json       # 只出 JSON（贴给 issue / AI）
    python tools/probe_accelerator.py --compile    # 额外跑 torch.compile 冒烟（慢，分钟级）

**设计约束**：
- stdlib + 可选 torch。venv 只有 pip 时也能跑完前半段（系统 / smi 段）。
- 每一项独立 try/except，**任何一项失败都不影响其余项** —— 探测脚本自己崩掉是最没用的
  结果。失败项在报告里记 `{"ok": false, "error": ...}`，保留异常类型与消息。
- 不写文件、不装包、不改环境变量，纯只读。

探测覆盖（对应本仓库真实用到的能力）：
  1. 系统 / Python / 容器
  2. smi 工具（nvidia-smi / hy-smi / rocm-smi / amd-smi）—— 驱动与卡型号
  3. torch build（cuda / hip 版本标签，区分 NVIDIA wheel 与 DTK wheel）
  4. 设备枚举（名称 / arch / 显存 / 计算能力）
  5. dtype 能力（bf16 / fp16 / fp8 存储与 cast —— Krea2 fp8 底模依赖）
  6. attention 后端（SDPA 三档 / flash_attn / xformers）
  7. 显存编排原语（Stream / Event / pinned / non_blocking / mem_get_info / OOM 类型）
  8. 相关 Python 包（triton / bitsandbytes / pynvml / onnxruntime EP）
  9. 加速器相关环境变量
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Callable

REPORT: dict[str, Any] = {}
_HUMAN: list[str] = []


def say(line: str = "") -> None:
    _HUMAN.append(line)


def head(title: str) -> None:
    say()
    say(f"── {title} " + "─" * max(0, 60 - len(title)))


def probe(key: str, fn: Callable[[], Any]) -> Any:
    """跑一个探测项，结果存进 REPORT[key]；异常吞掉并记录。

    返回值供后续项串联（拿不到时返回 None）。约定：fn 返回 dict 的话直接存，
    其余包一层 `{"value": ...}`，让 JSON 结构对消费方一致。
    """
    try:
        out = fn()
    except Exception as exc:  # noqa: BLE001  探测脚本必须自己不崩
        REPORT[key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return None
    if isinstance(out, dict):
        out.setdefault("ok", True)
        REPORT[key] = out
    else:
        REPORT[key] = {"ok": True, "value": out}
    return out


def run_cmd(args: list[str], timeout: int = 15) -> dict[str, Any]:
    """跑外部命令，返回 {rc, stdout, stderr}；不存在返回 {found: False}。"""
    exe = shutil.which(args[0])
    if not exe:
        return {"found": False}
    r = subprocess.run(
        [exe, *args[1:]],
        capture_output=True,
        text=True,
        timeout=timeout,
        errors="replace",
        check=False,
    )
    return {
        "found": True,
        "path": exe,
        "rc": r.returncode,
        "stdout": (r.stdout or "").strip(),
        "stderr": (r.stderr or "").strip(),
    }


# ── 1. 系统 / Python / 容器 ────────────────────────────────────────────
def probe_system() -> dict[str, Any]:
    """OS / 架构 / Python / 是否在容器里。

    容器判定看 /.dockerenv 与 cgroup —— DTK 镜像里跑是常态，报告里标出来能
    解释「宿主有卡但容器里看不到」这类问题（缺 --device=/dev/kfd 等挂载）。
    """
    in_docker = os.path.exists("/.dockerenv")
    cgroup = ""
    try:
        with open("/proc/1/cgroup", encoding="utf-8", errors="replace") as f:
            cgroup = f.read()
    except OSError:
        pass
    if not in_docker and ("docker" in cgroup or "containerd" in cgroup):
        in_docker = True

    # DCU 需要 /dev/kfd（HSA kernel driver）+ /dev/dri；容器没挂进来时
    # torch.cuda.is_available() 直接 False，这两行能立刻定位到挂载问题。
    dev_nodes = {
        p: os.path.exists(p) for p in ("/dev/kfd", "/dev/dri", "/dev/nvidiactl")
    }
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "python_exe": sys.executable,
        "in_container": in_docker,
        "dev_nodes": dev_nodes,
        "libc": platform.libc_ver()[1] if hasattr(platform, "libc_ver") else None,
    }


# ── 2. smi 工具 ────────────────────────────────────────────────────────
#: 各厂商的 smi 工具。海光 DTK 装 hy-smi（也可能同时有 rocm-smi，DTK 是 ROCm 分支）。
#: 探测顺序不代表优先级，全部都试 —— 报告要如实反映机器上有什么。
_SMI_PROBES: tuple[tuple[str, list[str]], ...] = (
    ("nvidia_smi", ["nvidia-smi", "--query-gpu=driver_version,name,memory.total",
                    "--format=csv,noheader"]),
    ("hy_smi", ["hy-smi"]),
    ("rocm_smi", ["rocm-smi"]),
    ("amd_smi", ["amd-smi", "static"]),
    ("hy_virtual", ["hy-virtual", "-show-device-info"]),
)


def probe_smi() -> dict[str, Any]:
    """逐个试各厂商 smi。

    重点是 **hy-smi 的实际输出格式** —— 显存 / 利用率监控要从这里解析，
    不同 DTK 版本的列名与分隔符有差异，必须看真机输出再写 parser。
    整段 stdout 原样保留（不截断），移植时按真实格式写解析。
    """
    out: dict[str, Any] = {}
    for key, args in _SMI_PROBES:
        try:
            out[key] = run_cmd(args)
        except Exception as exc:  # noqa: BLE001
            out[key] = {"found": True, "error": f"{type(exc).__name__}: {exc}"}
    return out


# ── 3. torch build ────────────────────────────────────────────────────
def probe_torch_build() -> dict[str, Any]:
    """torch 版本与它编译时绑定的 GPU 运行时。

    关键区分（本仓库当前代码在这里误判 DCU）：
    - NVIDIA wheel：`torch.version.cuda` = "12.8"、`torch.version.hip` = None
    - DTK / ROCm wheel：`torch.version.cuda` = **None**、`torch.version.hip` 有值
      （形如 "6.3.42134-..."），且 `torch.__version__` 常带 `+das.dtk...` 之类本地标签

    只看 `version.cuda is None` 就判「CPU-only 误装」会把 DCU 归错类 —— 必须同时看 hip。
    """
    import torch

    ver = torch.__version__
    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    if hip:
        kind = "hip"       # ROCm / 海光 DTK
    elif cuda:
        kind = "cuda"      # NVIDIA
    else:
        kind = "cpu"
    return {
        "torch_version": ver,
        "version_cuda": cuda,
        "version_hip": hip,
        "build_kind": kind,
        "is_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "torch_file": getattr(torch, "__file__", None),
        # ROCm build 上 torch.backends.cudnn 仍存在（HIP 映射到 MIOpen）；
        # 记下来确认 cudnn.benchmark 这类开关设了不会炸。
        "cudnn_available": bool(getattr(torch.backends, "cudnn", None)
                                and torch.backends.cudnn.is_available()),
        "cudnn_version": (torch.backends.cudnn.version()
                          if getattr(torch.backends, "cudnn", None)
                          and torch.backends.cudnn.is_available() else None),
    }


# ── 4. 设备枚举 ────────────────────────────────────────────────────────
def probe_devices() -> dict[str, Any]:
    """逐卡拿名称 / 显存 / arch。

    `gcnArchName` 只在 HIP build 上有（如 `gfx906` / `gfx928`），是判断 DCU 型号与
    kernel 兼容性的关键字段：flash-attn / AOTriton 的可用性按 gfx 号分档。
    """
    import torch

    if not torch.cuda.is_available():
        return {"devices": [], "note": "torch.cuda.is_available() == False"}
    devices = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free_total: Any
        try:
            free, total = torch.cuda.mem_get_info(i)
            free_total = {"free_gb": round(free / 1024**3, 2),
                          "total_gb": round(total / 1024**3, 2)}
        except Exception as exc:  # noqa: BLE001
            free_total = {"error": f"{type(exc).__name__}: {exc}"}
        devices.append({
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "capability": list(torch.cuda.get_device_capability(i)),
            "total_memory_gb": round(p.total_memory / 1024**3, 2),
            "multi_processor_count": getattr(p, "multi_processor_count", None),
            # HIP only；NVIDIA 上为 None
            "gcn_arch_name": getattr(p, "gcnArchName", None),
            "mem_get_info": free_total,
        })
    return {"devices": devices}


# ── 5. dtype 能力 ──────────────────────────────────────────────────────
def probe_dtypes() -> dict[str, Any]:
    """bf16 / fp16 / fp8 在设备上到底能不能用。

    fp8 是本项目最关键的不确定项：Krea2 用官方 fp8 权重当训练 / 推理底模，
    `quant_fp8.py` 的数值口径是 **存储 fp8 + 前向 dequant**：

        W_compute = W_fp8.to(input.dtype) [* scale.to(input.dtype)]

    也就是说只需要 fp8 的 **存储 + cast**，不需要 fp8 原生 matmul（`_scaled_mm`）。
    所以这里分开测三件事，让移植时知道能走到哪一步：
      - `e4m3fn_storage`：能不能在设备上放 fp8 张量
      - `e4m3fn_cast`：能不能 cast 回 bf16（**决定 fp8 底模可用性**）
      - `scaled_mm`：原生 fp8 matmul（本项目不依赖，仅记录）

    另测 `e4m3fnuz` —— ROCm 系某些架构的原生 fp8 变体是 fnuz（指数偏置不同）。
    若 fn 不可用而 fnuz 可用，说明 fp8 权重需要转换而非直接加载。
    """
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out: dict[str, Any] = {"device_used": dev}

    def _try(name: str, fn: Callable[[], Any]) -> None:
        try:
            out[name] = {"ok": True, "detail": fn()}
        except Exception as exc:  # noqa: BLE001
            out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # 基础计算 dtype：matmul 真跑一次，不只看 dtype 是否存在
    for dname, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16),
                         ("fp32", torch.float32)):
        def _mm(dt=dtype):
            a = torch.randn(64, 64, device=dev, dtype=dt)
            return float((a @ a).float().abs().mean())
        _try(f"{dname}_matmul", _mm)

    # autocast（训练用 mixed_precision）
    def _autocast():
        with torch.autocast(device_type="cuda" if dev == "cuda" else "cpu",
                            dtype=torch.bfloat16):
            a = torch.randn(32, 32, device=dev)
            return str((a @ a).dtype)
    _try("autocast_bf16", _autocast)

    out.update(_probe_fp8(dev))
    return out


def _probe_fp8(dev: str) -> dict[str, Any]:
    """fp8 三段式探测（存储 / cast / 原生 matmul）+ fnuz 变体。"""
    import torch

    res: dict[str, Any] = {}

    def _try(name: str, fn: Callable[[], Any]) -> None:
        try:
            res[name] = {"ok": True, "detail": fn()}
        except Exception as exc:  # noqa: BLE001
            res[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    for variant in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
        dtype = getattr(torch, variant, None)
        if dtype is None:
            res[f"{variant}_storage"] = {"ok": False, "error": "dtype 在此 torch 版本不存在"}
            continue

        def _storage(dt=dtype):
            t = torch.randn(64, 64, device=dev).to(dt)
            return {"shape": list(t.shape), "dtype": str(t.dtype),
                    "nbytes": t.untyped_storage().nbytes()}
        _try(f"{variant}_storage", _storage)

        # 这一项直接对应 quant_fp8._fp8_linear_forward 的真实路径：
        # fp8 权重 cast 到 bf16 后乘 scale，再走 F.linear。
        def _dequant_linear(dt=dtype):
            w = torch.randn(128, 64, device=dev).to(dt)
            scale = torch.tensor(0.5, device=dev, dtype=torch.float32)
            x = torch.randn(8, 64, device=dev, dtype=torch.bfloat16)
            w_c = w.to(x.dtype) * scale.to(x.dtype)
            y = torch.nn.functional.linear(x, w_c)
            return {"out_dtype": str(y.dtype), "out_mean": float(y.float().abs().mean())}
        _try(f"{variant}_dequant_linear", _dequant_linear)

    # 原生 fp8 matmul —— 本项目不走这条路（Comfy parity 用 dequant），仅记录
    def _scaled_mm():
        a = torch.randn(64, 64, device=dev).to(torch.float8_e4m3fn)
        b = torch.randn(64, 64, device=dev).to(torch.float8_e4m3fn).t()
        s = torch.tensor(1.0, device=dev)
        y = torch._scaled_mm(a, b, scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
        return {"out_dtype": str(y.dtype)}
    _try("scaled_mm_native", _scaled_mm)
    return res


# ── 6. attention 后端 ─────────────────────────────────────────────────
def probe_attention() -> dict[str, Any]:
    """SDPA 三档后端 + flash_attn + xformers。

    本项目 `attention_backend` 有三档（flash_attn / xformers / none=SDPA）。SDPA 是
    保底路径，所以先确认它在此设备上走的是哪个 kernel：ROCm 上 flash / mem-efficient
    后端依赖 AOTriton，老版本只有 math 后端（能跑但慢且省不了显存）。

    每个 SDPA 后端用 `sdpa_kernel` 单独强制开一次实测 —— 只查 `can_use_*` 这类
    静态标志不够，真跑才知道有没有 kernel。
    """
    import torch
    import torch.nn.functional as F

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out: dict[str, Any] = {"device_used": dev}
    q = torch.randn(2, 8, 512, 64, device=dev, dtype=torch.bfloat16)

    def _try(name: str, fn: Callable[[], Any]) -> None:
        try:
            out[name] = {"ok": True, "detail": fn()}
        except Exception as exc:  # noqa: BLE001
            out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    _try("sdpa_default", lambda: {
        "out_mean": float(F.scaled_dot_product_attention(q, q, q).float().abs().mean()),
    })

    # 逐后端强制。torch 2.9 用 torch.nn.attention.sdpa_kernel + SDPBackend 枚举。
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        backends = {
            "flash": SDPBackend.FLASH_ATTENTION,
            "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math": SDPBackend.MATH,
        }
        for bname, backend in backends.items():
            def _forced(b=backend):
                with sdpa_kernel(b):
                    o = F.scaled_dot_product_attention(q, q, q)
                return {"out_mean": float(o.float().abs().mean())}
            _try(f"sdpa_{bname}", _forced)
    except ImportError as exc:
        out["sdpa_kernel_api"] = {"ok": False, "error": str(exc)}

    out.update(_probe_attn_packages(dev))
    return out


def _probe_attn_packages(dev: str) -> dict[str, Any]:
    """flash_attn / xformers：装没装 + 真调一次能不能出结果。

    「装上了但一调就崩」在 ROCm 系很常见（wheel 是给别的 gfx 架构编的），所以
    不能只看 import 成功。调用形状对齐本项目实际用法：
    - flash_attn：`flash_attn_func(q, k, v)`，[B, S, H, D] 布局
    - xformers：`memory_efficient_attention`，同 BSHD
    """
    import torch

    res: dict[str, Any] = {}
    q = torch.randn(2, 512, 8, 64, device=dev, dtype=torch.bfloat16)

    try:
        import flash_attn
        ver = getattr(flash_attn, "__version__", "?")
        try:
            from flash_attn import flash_attn_func
            o = flash_attn_func(q, q, q)
            res["flash_attn"] = {"ok": True, "version": ver,
                                 "call_ok": True,
                                 "out_mean": float(o.float().abs().mean())}
        except Exception as exc:  # noqa: BLE001
            res["flash_attn"] = {"ok": True, "version": ver, "call_ok": False,
                                 "error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001
        res["flash_attn"] = {"ok": False, "installed": False,
                             "error": f"{type(exc).__name__}: {exc}"}

    try:
        import xformers
        import xformers.ops as xops
        ver = getattr(xformers, "__version__", "?")
        try:
            o = xops.memory_efficient_attention(q, q, q)
            res["xformers"] = {"ok": True, "version": ver, "call_ok": True,
                               "out_mean": float(o.float().abs().mean())}
        except Exception as exc:  # noqa: BLE001
            res["xformers"] = {"ok": True, "version": ver, "call_ok": False,
                               "error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001
        res["xformers"] = {"ok": False, "installed": False,
                           "error": f"{type(exc).__name__}: {exc}"}
    return res


# ── 7. 显存编排原语 ───────────────────────────────────────────────────
def probe_memory_primitives() -> dict[str, Any]:
    """Block swap 依赖的整套原语，逐个真跑。

    `runtime/training/block_swap.py` 用的是：独立 copy Stream + Event 跨流同步 +
    pinned 内存 + `non_blocking=True` H2D 拷贝。这套在 HIP 上语义相同，但必须
    实测确认 —— 其中任一项不灵，block swap（16GB 卡跑 Krea2 的前提）就不能开。

    另测 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`（anima_train.py 顶层设了它）
    与 OOM 异常类型（多处 `except torch.cuda.OutOfMemoryError`）。
    """
    import torch

    out: dict[str, Any] = {}
    if not torch.cuda.is_available():
        return {"skipped": "torch.cuda.is_available() == False"}

    def _try(name: str, fn: Callable[[], Any]) -> None:
        try:
            out[name] = {"ok": True, "detail": fn()}
        except Exception as exc:  # noqa: BLE001
            out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _stream_event_pinned():
        """完整复刻 block_swap 的搬运模式：pinned CPU → 独立流异步上卡 → event 同步。"""
        src = torch.randn(1024, 1024, dtype=torch.bfloat16).pin_memory()
        dst = torch.empty_like(src, device="cuda")
        copy_stream = torch.cuda.Stream()
        ready = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            dst.copy_(src, non_blocking=True)
            ready.record(copy_stream)
        torch.cuda.current_stream().wait_event(ready)
        torch.cuda.synchronize()
        return {"is_pinned": src.is_pinned(),
                "match": bool(torch.equal(dst.cpu(), src))}
    _try("stream_event_pinned_copy", _stream_event_pinned)

    _try("empty_cache", lambda: (torch.cuda.empty_cache(), "ok")[1])
    _try("memory_stats", lambda: {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    })
    _try("reset_peak_memory_stats",
         lambda: (torch.cuda.reset_peak_memory_stats(), "ok")[1])
    _try("oom_error_type", lambda: {
        "has_torch_cuda_OutOfMemoryError": hasattr(torch.cuda, "OutOfMemoryError"),
        "qualname": getattr(getattr(torch.cuda, "OutOfMemoryError", None),
                            "__qualname__", None),
    })
    _try("rng_state", lambda: {
        "get_set_ok": (torch.cuda.set_rng_state(torch.cuda.get_rng_state()), True)[1],
    })
    _try("alloc_conf_env", lambda: {
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "PYTORCH_HIP_ALLOC_CONF": os.environ.get("PYTORCH_HIP_ALLOC_CONF"),
    })
    return out


# ── 8. 相关 Python 包 ─────────────────────────────────────────────────
#: 本项目会 import / 装的加速相关包。`bitsandbytes` 是可选 8-bit 优化器；
#: `triton` 服务 torch.compile 与部分 SDPA 后端；`pynvml` 是 topbar 显存监控。
_PKGS: tuple[str, ...] = (
    "triton", "pytorch_triton_rocm", "bitsandbytes", "pynvml", "onnxruntime",
    "diffusers", "transformers", "accelerate", "peft", "safetensors",
    "lycoris_lora", "prodigyopt", "optimum", "spandrel",
)


def probe_packages() -> dict[str, Any]:
    """逐个 import 拿版本；顺带把 onnxruntime 的可用 EP 列出来。

    onnxruntime 的 EP 列表决定 WD14 / CLTagger 打标能否上卡。海光侧一般是
    MIGraphX EP 或只有 CPU EP，与 NVIDIA 的 CUDAExecutionProvider 不同名，
    所以 `current_runtime()` 里那套 `cuda_available` 判定在 DCU 上恒为 False。
    """
    import importlib
    import importlib.metadata as md

    out: dict[str, Any] = {}
    for name in _PKGS:
        try:
            out[name] = {"installed": True, "version": md.version(name)}
        except Exception:  # noqa: BLE001  含 PackageNotFoundError
            out[name] = {"installed": False}

    try:
        ort = importlib.import_module("onnxruntime")
        out["onnxruntime_providers"] = {
            "version": getattr(ort, "__version__", None),
            "available": list(ort.get_available_providers()),
        }
    except Exception as exc:  # noqa: BLE001
        out["onnxruntime_providers"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


# ── 9. 环境变量 ───────────────────────────────────────────────────────
#: 影响加速器选择 / 行为的环境变量。DCU 侧 `HIP_VISIBLE_DEVICES` 与
#: `CUDA_VISIBLE_DEVICES` 都可能生效；`HSA_OVERRIDE_GFX_VERSION` 是 ROCm 系
#: 常用的架构伪装开关（wheel 没给你的 gfx 编 kernel 时用），排错时必须看到。
_ENV_KEYS: tuple[str, ...] = (
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL", "HSA_OVERRIDE_GFX_VERSION", "PYTORCH_ROCM_ARCH",
    "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF",
    "TORCH_BLAS_PREFER_HIPBLASLT", "MIOPEN_USER_DB_PATH",
    "XFORMERS_FORCE_DISABLE_TRITON", "LORA_RAM_GUARD",
    "ROCM_PATH", "HIP_PATH", "DTK_PATH", "CUDA_HOME", "CUDA_PATH",
    "LD_LIBRARY_PATH",
)


def probe_env() -> dict[str, Any]:
    return {k: os.environ.get(k) for k in _ENV_KEYS}


def probe_compile() -> dict[str, Any]:
    """torch.compile 冒烟（--compile 才跑；首次编译分钟级）。

    本项目当前不主动 torch.compile 模型，但 inductor 能否工作是后续优化的前提，
    也是 triton 在此平台是否健全的强信号。
    """
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.compile
    def f(x):
        return (x * 2 + 1).sin().sum()

    x = torch.randn(256, 256, device=dev)
    return {"result": float(f(x)), "device_used": dev}


# ── 人类可读摘要 ──────────────────────────────────────────────────────
def _mark(ok: bool | None) -> str:
    """三态标记：True=可用 / False=不可用 / None=未探测。"""
    return {True: "[ok]  ", False: "[FAIL]", None: "[--]  "}[ok]


def _sub_ok(section: str, key: str) -> bool | None:
    """读 REPORT[section][key]['ok']；缺失返回 None。"""
    sec = REPORT.get(section) or {}
    item = sec.get(key)
    if not isinstance(item, dict):
        return None
    return bool(item.get("ok"))


def render_summary() -> None:
    """把最影响决策的结论摘出来，避免用户在几百行 JSON 里找。"""
    build = REPORT.get("torch_build") or {}
    devs = (REPORT.get("devices") or {}).get("devices") or []

    head("结论摘要")
    kind = build.get("build_kind")
    vendor = {"hip": "ROCm / 海光 DTK", "cuda": "NVIDIA CUDA", "cpu": "CPU-only"}.get(
        kind, "未知")
    say(f"torch build       : {build.get('torch_version')}  → {vendor}")
    say(f"  version.cuda    : {build.get('version_cuda')}")
    say(f"  version.hip     : {build.get('version_hip')}")
    say(f"  is_available    : {build.get('is_available')}   设备数 {build.get('device_count')}")
    for d in devs:
        arch = f"  arch={d.get('gcn_arch_name')}" if d.get("gcn_arch_name") else ""
        say(f"  [{d['index']}] {d['name']}  {d['total_memory_gb']}GB{arch}")

    say()
    say("训练关键能力：")
    say(f"  {_mark(_sub_ok('dtypes', 'bf16_matmul'))} bf16 matmul（训练主精度）")
    say(f"  {_mark(_sub_ok('dtypes', 'autocast_bf16'))} autocast bf16（mixed_precision）")
    say(f"  {_mark(_sub_ok('dtypes', 'float8_e4m3fn_dequant_linear'))} "
        f"fp8 e4m3fn dequant linear（Krea2 fp8 底模）")
    say(f"  {_mark(_sub_ok('attention', 'sdpa_default'))} SDPA 默认 dispatch")
    say(f"  {_mark(_sub_ok('attention', 'sdpa_math'))} SDPA math 后端（保底，无外部依赖）")
    say(f"  {_mark(_sub_ok('attention', 'sdpa_flash'))} SDPA flash 后端（快，需 flash-attn 包）")
    say(f"  {_mark(_sub_ok('attention', 'sdpa_mem_efficient'))} SDPA mem-efficient 后端")
    fa = (REPORT.get("attention") or {}).get("flash_attn") or {}
    say(f"  {_mark(bool(fa.get('call_ok')))} flash_attn 包可调用"
        f"（installed={fa.get('ok', False)}, version={fa.get('version')}）")
    xf = (REPORT.get("attention") or {}).get("xformers") or {}
    say(f"  {_mark(bool(xf.get('call_ok')))} xformers 可调用"
        f"（installed={xf.get('ok', False)}, version={xf.get('version')}）")
    say(f"  {_mark(_sub_ok('memory', 'stream_event_pinned_copy'))} "
        f"Stream+Event+pinned 异步搬运（block swap 前提）")

    # SDPA 全线失败 / 只剩 math 时给出明确结论 —— 这是最容易被误读的一段：
    # 「sdpa_default 失败」看着像致命错误，实际本项目启动期会关掉坏后端退到 math。
    if _sub_ok("attention", "sdpa_math") and not _sub_ok("attention", "sdpa_flash"):
        say()
        say("  → flash 后端不可用但 math 可用：训练能跑（启动期自动关掉坏后端），")
        say("    但更慢、长序列更吃显存。装 DTK 渠道的 flash-attn 可恢复快路径，")
        say("    先跑 `bash tools/find_flash_attn.sh` 看本机有没有现成包。")
    elif not _sub_ok("attention", "sdpa_math"):
        say()
        say("  → 连 math 后端都不可用：attention 无可用实现，训练无法进行。")
        say("    这不正常，请把本报告完整贴出来。")

    smi = REPORT.get("smi") or {}
    found = [k for k, v in smi.items() if isinstance(v, dict) and v.get("found")]
    say()
    say(f"smi 工具          : {', '.join(found) if found else '（一个都没找到）'}")
    ort = (REPORT.get("packages") or {}).get("onnxruntime_providers") or {}
    say(f"onnxruntime EP    : {ort.get('available', ort.get('error'))}")


def main() -> int:
    json_only = "--json" in sys.argv
    do_compile = "--compile" in sys.argv

    probe("system", probe_system)
    probe("smi", probe_smi)

    # torch 探测全部依赖 import torch 成功；失败时后续项跳过（REPORT 里留错误）
    try:
        import torch  # noqa: F401
        has_torch = True
    except Exception as exc:  # noqa: BLE001
        REPORT["torch_build"] = {"ok": False,
                                 "error": f"import torch 失败: {type(exc).__name__}: {exc}"}
        has_torch = False

    if has_torch:
        probe("torch_build", probe_torch_build)
        probe("devices", probe_devices)
        probe("dtypes", probe_dtypes)
        probe("attention", probe_attention)
        probe("memory", probe_memory_primitives)
        probe("packages", probe_packages)
        if do_compile:
            probe("compile", probe_compile)
    probe("env", probe_env)

    payload = json.dumps(REPORT, indent=2, ensure_ascii=False, default=str)
    if json_only:
        print(payload)
        return 0

    if has_torch:
        render_summary()
    print("\n".join(_HUMAN))
    print()
    print("=" * 68)
    print("完整 JSON（把下面整段贴回去）：")
    print("=" * 68)
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
