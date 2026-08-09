"""xformers 安装服务（简化版，类比 flash_attention_setup）。

xformers 与 flash_attn 同为 attention 加速 C extension，但安装路径**显著简单**：
  - flash_attn：依赖 dao-AILab + mjun0812 prebuild 的 GitHub Releases，
    每个 torch+cuda+python 组合一个 wheel，需要解析 wheel 名 + 评分匹配
  - xformers：facebook 官方 PyPI 直接发 wheel，与 torch+cuda 强绑定但
    PyTorch 官方 wheel index (download.pytorch.org/whl/cuXXX) 已经
    把对应 cu_tag 的 wheel 集中起来。

所以本服务只暴露：
  - current_status() → {installed, version}
  - install() → pip install xformers --index-url <torch-cuda-index>

不复刻 flash_attention_setup 的 GitHub Releases 解析 / 候选列表 UI。
装失败时把 stderr 透传，让用户自己看（多数失败 = 上游没出对应 torch+cu
组合的 wheel，需要换 torch 版本或等上游覆盖）。

**后端边界**：xformers 官方只发 CUDA build wheel，海光 DCU（DTK / HIP）上装不了
也用不了 —— DCU 的正常路径是 PyTorch SDPA，**不是**「降级到不可用」：SDPA 在
DTK 上有 HIP kernel 实现，attention 该有的性能照常拿得到，只是不走 xformers 那套
cutlass kernel。后端一律问 ``utils.accelerator``（不自己读 ``torch.version.hip``），
拦截闸见 :func:`_xformers_blocked` —— 只拦确认是 DCU 的情况，``cuda`` 与
「torch 还没装好」的 ``cpu`` 不确定态都保持旧行为。
"""
from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from typing import Any, Optional

from utils import accelerator


def _pip_install_blocked() -> bool:
    """``pip install xformers`` 是否该在当前后端上拦截。**只有确认是海光 DCU 时**为 True。

    拦的是**自动安装**，不是 xformers 本身 —— 海光在光合社区发布配套 wheel（如
    ``xformers-0.0.33+das.opt1.dtk2604.torch251``），装上后功能完全可用。公开源上
    没有 DCU wheel，`pip install xformers` 只会拉到 CUDA build 或干脆找不到，所以
    自动装这条路要拦；手动装那条路要引导。

    闸门刻意写成「只拦确定不行的」而不是「只放确定行的」（``backend == "cuda"``）——
    因为那样会把 ``cpu`` 后端一起拦掉，而 ``cpu`` 在本项目里是**不确定态**，不等于
    「确认这机器没 GPU」：``accelerator.detect()`` 在 torch 还没装（venv 首装阶段）
    或 torch import 失败（Windows 缺 DLL、WinError 126）时一律报 ``cpu``。NVIDIA
    机器首装途中正好处在这个状态，用「只放 cuda」当闸会让它被拒装 xformers、还收到
    一段讲 DCU 的文案 —— 属于 NVIDIA 路径回归，是本次双后端移植的头号禁忌。

    所以：dcu 拦，其余（cuda / cpu / 探测抛异常）一律放行 = 旧行为逐字节不变。
    """
    try:
        return accelerator.is_dcu()
    except Exception:  # noqa: BLE001
        return False


def current_status() -> dict[str, Any]:
    """xformers 当前安装状态：{installed: bool, version: str|None}。"""
    try:
        version = importlib.metadata.version("xformers")
        return {"installed": True, "version": version}
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}


def detect_attention_backend() -> str:
    """根据当前装了什么决定 attention backend。
    优先级 flash_attn > xformers > none（PyTorch SDPA）。
    给 secrets.generate.attention_backend='auto' 时用。

    **不按后端跳过 xformers**：海光在光合社区发布 DCU 配套 wheel，装上后功能可用，
    所以判据是「装了没有」而不是「是什么卡」。DCU 上没装 xformers 时自然落到 none
    （= SDPA），与之前的效果一致，但装了的用户不再被无谓地排除。
    """
    try:
        importlib.metadata.version("flash_attn")
        return "flash_attn"
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        importlib.metadata.version("xformers")
        return "xformers"
    except importlib.metadata.PackageNotFoundError:
        pass
    return "none"


def disable_triton_probe(env: dict[str, str]) -> None:
    """替 xformers 短路 triton 探测（子进程 env 注入用）。

    xformers 启用后其 `_is_triton_available()` 会 `import triton`。triton 官方
    不发 Windows wheel，未安装时必然 ImportError，xformers 用
    `logger.warning(..., exc_info=True)` 把完整 traceback 打进 task log ——
    还会被失败摘要 `_tail_log_for_error_msg`（取最后一处 Traceback）误当
    失败原因展示给用户。

    无条件短路：xformers 里 triton 只服务 LLM 型 kernel（fmha triton_splitk /
    rmsnorm / rope_padded / tiled_matmul），本 app 的 memory_efficient_attention
    （含 NaViT varlen）走 cutlass/flash C++ kernel，triton 装没装、好没好都
    零参与，探测结果与 warning 对用户均无价值。torch.compile 的 triton 使用
    不读本变量，不受影响。`XFORMERS_FORCE_DISABLE_TRITON=1` 在 xformers 源码
    里的检查位于 `import triton` 之前，设了就完全跳过探测；setdefault 保证
    用户显式设过的值优先，且 xformers 侧 `XFORMERS_ENABLE_TRITON=1` 优先级
    更高，仍是强开逃生口。

    **DCU 上刻意不加后端分支、保持无条件注入**，两个理由：
    1. 这个变量只被 xformers 自己读，而 DCU 上 xformers 装不上（install() 直接拒），
       所以在 DCU 上它是纯无害的死变量 —— 加个 if 只会多一处后端分支要维护。
    2. DTK 自带 triton，但 torch.compile 走的是 torch 内部的 triton 集成，**不读**
       这个变量（见上文），设了不影响任何 DCU 上真实使用 triton 的路径。
    """
    env.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")


def _torch_cuda_index() -> Optional[str]:
    """从 `torch.__version__` 的 `+cuXXX` 后缀推 PyTorch CUDA index URL。

    xformers wheel 与 torch ABI 强绑定（每个 xformers 版本锁定特定 torch+cuda），
    必须装与当前 torch 同 CUDA 的 wheel。PyTorch 官方 index 按 cu_tag 分组：
        https://download.pytorch.org/whl/cu128
        https://download.pytorch.org/whl/cu130
        ...

    ABI 检测原则与 flash_attention_setup.detect_env() 一致：从 torch 拿，
    不从 nvidia-smi 拿（nvidia-smi 是 driver 支持的 CUDA，不是 PyTorch 编译的）。

    DCU 上返回 None 是**正确且不需要额外分支**的：DTK wheel 的版本串形如
    `2.9.0+das.opt...dtk...`，没有 `+cuNNN` 片段，正则天然不匹配。这里不加后端
    判断是刻意的——真正的拦截在 install()（DCU 直接拒），走不到本函数；万一将来
    有人绕过 install() 复用它，返回 None 也只是退到 PyPI 默认源，不会拼出一个
    错误的 CUDA index URL。
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    m = re.search(r"\+cu(\d+)", torch.__version__)
    if m:
        return f"https://download.pytorch.org/whl/cu{m.group(1)}"
    return None


def install() -> dict[str, Any]:
    """pip install xformers，自动按当前 torch 的 CUDA index 选 wheel。

    返回 {installed, version, stdout_tail, restart_required}。
    安装失败抛 RuntimeError，message 含 stderr 末尾（多数 wheel 找不到时
    pip 会打印「No matching distribution found for xformers」）。

    `restart_required=True` 因为 xformers 是 C extension —— 装好后必须重启
    Studio 进程才能 import（与 flash_attn 同）。

    DCU 上拒绝**自动安装**（不是拒绝 xformers 本身）—— 见函数体内说明。
    """
    # DCU 上拦自动安装：公开源上没有 DCU wheel，`pip install xformers` 只会拉到
    # CUDA build（.so 链接 libcudart / libcublas，import 必挂）或直接找不到。
    #
    # 但 xformers 在 DCU 上**不是不可用** —— 海光在光合社区发布配套 wheel，与 DTK /
    # torch 版本严格配套（如 xformers-0.0.33+das.opt1.dtk2604.torch251）。装上后本项目
    # 会自动识别并使用（detect_attention_backend 只看装了没有）。所以这里的文案要
    # 引导「去哪手动装」，而不是说「你的卡不支持」。
    if _pip_install_blocked():
        try:
            vendor = accelerator.detect().vendor_label
        except Exception:  # noqa: BLE001
            vendor = accelerator.VENDOR_LABEL["dcu"]
        raise RuntimeError(
            f"当前后端是 {vendor}，不能自动安装 xformers。\n"
            "公开源（PyPI / PyTorch index）上只有 CUDA build 的 xformers wheel，"
            "装到 DCU 上 import 会直接失败。\n"
            "海光有配套 wheel，走光合开发者社区单独发布，需与镜像的 DTK 版本和 torch "
            "版本都对齐（文件名形如 xformers-0.0.33+das.opt1.dtk2604.torch251-py3-none-any.whl）：\n"
            "  1. 从光合开发者社区 / DTK 配套仓库取匹配的 wheel\n"
            "  2. pip install <本地 wheel 路径>\n"
            "  3. 重启 Studio（C extension 不能热替换），本页会显示版本\n"
            "装好后训练 / 出图会自动识别使用，无需再改设置。\n"
            "不装也能训练：attention 会走 PyTorch SDPA。"
        )

    cmd = [sys.executable, "-m", "pip", "install", "xformers"]
    index = _torch_cuda_index()
    if index:
        cmd += ["--index-url", index]

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("pip install xformers 超时（10 分钟）") from exc

    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-1500:]
        raise RuntimeError(
            f"pip install xformers 失败 (exit {r.returncode}):\n{tail}"
        )

    status = current_status()
    return {
        **status,
        "stdout_tail": (r.stdout or "")[-1500:],
        "restart_required": True,
    }
