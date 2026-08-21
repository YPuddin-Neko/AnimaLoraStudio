"""PyTorch 装包检测 + 一键重装服务（PR-S2 / PR-S2.1）。

为什么单立一个 service：
- requirements.txt 写 `torch>=2.0.0` 不指 `--index-url`，pip 默认装 CPU wheel —— 用户
  有 NVIDIA GPU 的常态是 PR-4 启动期警告，但**已有 venv** 用户不会被自动修。
- onnxruntime_setup 已有现成的 detect_cuda（nvidia-smi 探针）+ pip helper 模式，本
  service 同构：detect_torch / recommend_index_url / reinstall。
- UI（Settings → 训练 → PyTorch section）和 CLI 都走这套 API，单一 source of truth。

设计要点：
- `pip uninstall torch torchvision -y && pip install torch torchvision --index-url <cu>`
  —— 不带 `--upgrade`；显式 reinstall 强制走指定 index-url
- 驱动版本 → cu wheel 映射保守取「该驱动能跑的最高 cu」（NVIDIA 向下兼容文档）
- timeout 30 分钟（torch + cuda 依赖 ~3 GB，慢网常见）

**后端边界（海光 DCU 移植）**：本 service 的整套「卸装重装」能力**只对 NVIDIA
成立**。海光 DTK 的 torch wheel 由厂商镜像预装、不在 PyPI 上，pip 覆盖会直接
报废环境（装不回来，用户只能重建容器）。所以 DCU 上：`reinstall()` /
`_decide_target_tag()` 抛 RuntimeError 拒绝执行，`current_status()` 回
`can_manage_torch_install=False` 让 UI 置灰按钮。判定一律问
`utils/accelerator.py`，不在本文件自己读 `torch.version.hip`。
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import sysconfig
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Any, Optional

from utils import accelerator

from . import onnxruntime as onnxruntime_setup

logger = logging.getLogger(__name__)

# PyTorch 公布的 wheel index URLs。https://download.pytorch.org/whl/<tag>
# 顺序：从新到老，auto 选第一个驱动支持的。新 cu 加进来时插在列首。
SUPPORTED_INDEX_TAGS: tuple[str, ...] = ("cu128", "cu126", "cu124", "cu118", "cpu")

# 驱动版本 -> 该驱动能跑的 PyTorch CUDA wheel tag。NVIDIA 驱动向下兼容（新驱动跑老 cu）。
# 阈值取 NVIDIA 官方 CUDA Toolkit Release Notes 的「Driver Required」。
# 来源：https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id5
_DRIVER_TO_BEST_CU: tuple[tuple[float, str], ...] = (
    (555.0, "cu128"),  # CUDA 12.8 → driver R555+
    (550.0, "cu126"),  # CUDA 12.6 → driver R550+ (12.6 ≈ 同档)
    (545.0, "cu124"),  # CUDA 12.4 → driver R545+
    (470.0, "cu118"),  # CUDA 11.8 → driver R470+
)

PYPI_INDEX_BASE = "https://download.pytorch.org/whl"

#: DCU 上 `cuda_build` 取这个值。DTK wheel 的 `torch.__version__` 形如
#: `2.9.0+das.dtk2604`，既不是 `cu\d+` 也不是 `cpu`，用厂商工具链名标识最直白。
#: 前端只用它做展示 / 高亮，重装按钮由 `can_manage_torch_install` 单独管。
DTK_BUILD_TAG = "dtk"

#: DCU 上拒绝重装时的说明。抽成常量供 `reinstall()` / `_decide_target_tag()` /
#: `current_status()` 共用一份措辞（UI 和 CLI 看到的原因要一致）。
DCU_REFUSE_REASON = (
    "当前是海光 DCU (DTK) 环境，本项目不接管 PyTorch 安装。"
    "DTK 版 torch 由厂商镜像预装，wheel 不在 PyPI 上，"
    "任何 pip install/uninstall torch 都会把它替换成 PyPI 的 CPU 版或 NVIDIA 版，"
    "环境会直接报废且无法用 pip 装回来（只能重建容器）。"
    "要换 torch 版本请换用对应 DTK 版本的官方镜像。"
)


def _index_url_for(tag: str) -> Optional[str]:
    """`cu128` → `https://download.pytorch.org/whl/cu128`；`cpu` → 同；非法 → None。"""
    if tag not in SUPPORTED_INDEX_TAGS:
        return None
    return f"{PYPI_INDEX_BASE}/{tag}"


def can_manage_torch_install() -> bool:
    """本进程是否可以替用户装 / 重装 torch。`utils.accelerator` 策略的薄壳。

    存在的唯一理由是**避免为了判后端而 import torch**：
    `accelerator.should_manage_torch_install()` 内部走 `detect()` → `import torch`，
    而本 service 的调用点（`pending_install.apply_pending` → `reinstall`）明确要求
    「在任何 import torch 之前跑」—— Windows 上 torch 的 `.pyd` 一旦被本进程加载，
    pip uninstall 就撞 `[WinError 5]` 并在 site-packages 留 `~orch` 僵尸目录
    （见 `_cleanup_zombie_dirs` 与 `pending_install` 顶部注释）。

    所以分两条路，都问 `utils/accelerator.py`，不自己判 `torch.version.hip`：
    - torch 已在本进程 import 过 → 直接问权威 API（不产生新的 import 副作用）
    - 还没 import → 用纯 stdlib 的 `probe_stdlib()`（跑 smi / 看 /dev/kfd）
    """
    if "torch" in sys.modules:
        return accelerator.should_manage_torch_install()
    return accelerator.probe_stdlib().backend != "dcu"


def recommend_cu_tag(driver_version: Optional[str]) -> str:
    """根据 NVIDIA 驱动版本返回推荐 cu tag；驱动太旧 / 没驱动 → 'cpu'。

    **仅对 NVIDIA 有意义**：入参是 nvidia-smi 报的驱动版本，出参是 PyTorch 官方
    CUDA wheel 的 index tag。DCU 上不要调它（调用方先过
    :func:`can_manage_torch_install`），否则会给出「装 cpu wheel」这种破坏性建议。
    """
    if not driver_version:
        return "cpu"
    try:
        major_minor = float(".".join(driver_version.split(".")[:2]))
    except (ValueError, AttributeError):
        return "cpu"
    for threshold, tag in _DRIVER_TO_BEST_CU:
        if major_minor >= threshold:
            return tag
    return "cpu"


def detect_torch() -> dict[str, Any]:
    """读 dist-info + import 后探针，返回当前 torch 状态。

    `cuda_build`：
    - 'cu128' / 'cu126' / 'cu124' / 'cu118' —— PyTorch CUDA wheel
    - 'dtk' —— 海光 DTK wheel（HIP build，:data:`DTK_BUILD_TAG`）
    - 'cpu' —— CPU-only wheel（既无 version.cuda 也无 version.hip）
    - None —— torch 未装

    `cuda_available` 表示 `torch.cuda.is_available()` —— 装了 CUDA wheel 也可能因驱动 /
    WSL 问题为 False。DCU 上这个名字照用（HIP build 复用整套 `torch.cuda` API），
    False 通常意味着容器没挂 /dev/kfd 或 DTK 装得不全。

    历史 bug：原实现只用 `re.search(r"\\+(cu\\d+|cpu)$", torch.__version__)` +
    `torch.version.cuda is None` 判 build，DTK wheel（版本形如 `2.9.0+das.dtk2604`、
    `version.cuda` 恒为 None）会被判成 `cuda_build='cpu'`，连带触发「检测到 GPU 但装了
    CPU 版 torch」误报和「重装 cu128」这种对 DTK 环境**破坏性**的建议。现在后端一律
    问 `utils/accelerator.py`。

    新增字段（`backend` / `vendor_label` / `cuda_version` / `hip_version`）是加法，
    原有 5 个 key 语义不变 —— 下游有 `/api/torch/status` 端点、cli.py、前端。
    """
    # 先问后端（进程内缓存的权威事实），再补 dist-info 视角的安装状态。
    info = accelerator.detect()
    backend_fields: dict[str, Any] = {
        "backend": info.backend,
        "vendor_label": info.vendor_label,
        "cuda_version": info.cuda_version,
        "hip_version": info.hip_version,
    }

    try:
        installed_version = _pkg_version("torch")
    except PackageNotFoundError:
        return {
            "installed": False,
            "version": None,
            "cuda_build": None,
            "cuda_available": False,
            "device_name": None,
            **backend_fields,
        }

    cuda_build: Optional[str] = None
    cuda_available = False
    device_name: Optional[str] = None
    try:
        import torch  # type: ignore[import-not-found]  # noqa: PLC0415
        if info.backend == "dcu":
            cuda_build = DTK_BUILD_TAG
        else:
            # 以下是 wheel tag 的**字符串推导**，不是后端判定（后端已由上面的
            # accelerator.detect() 定了，DTK 走不到这里）。所以照旧读
            # torch.version.cuda —— 逐字保持 NVIDIA 路径原有行为。
            # torch.__version__ 形如 "2.5.0+cu128" / "2.5.0+cpu" / "2.5.0"
            m = re.search(r"\+(cu\d+|cpu)$", torch.__version__)
            if m:
                cuda_build = m.group(1)
            else:
                # 兼容旧 build 没 + 后缀的情况，靠 torch.version.cuda
                cuda_v = getattr(torch.version, "cuda", None)
                if cuda_v is None:
                    cuda_build = "cpu"
                else:
                    # cuda_v 形如 "12.8"；映射到 wheel tag
                    clean = cuda_v.replace(".", "")
                    cuda_build = f"cu{clean}"
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            try:
                device_name = torch.cuda.get_device_name(0)
            except Exception:  # noqa: BLE001
                device_name = "?"
    except ImportError:
        pass

    return {
        "installed": True,
        "version": installed_version,
        "cuda_build": cuda_build,
        "cuda_available": cuda_available,
        "device_name": device_name,
        **backend_fields,
    }


def current_status() -> dict[str, Any]:
    """打包给 UI 用：torch 状态 + 驱动检测 + 推荐 cu tag + 诊断 flag。

    两个误装诊断 flag（`is_cpu_with_gpu` / `is_cuda_build_unavailable`）语义是
    **NVIDIA 口径**，DCU 上按后端 gate 掉恒为 False：

    - `is_cpu_with_gpu` 的证据是 nvidia-smi 探到卡（`cuda_detect.available`），
      DCU 机器上没有 nvidia-smi，本来就不会命中；显式 gate 是为了「同机既有 N 卡
      又有 DCU」这种畸形环境不误报。
    - `is_cuda_build_unavailable` 会命中：DTK build 的 `cuda_build='dtk'` 不在
      `(None, 'cpu')` 里，容器忘挂 /dev/kfd 时 `cuda_available=False`。但它的 UI
      文案是「NVIDIA 驱动 / WSL 问题」，对 DCU 是误导 —— 改由新增的
      `is_backend_unavailable` 承接（后端中立），前端按 backend 选文案。

    新增字段（加法，不动原有 key）：
    - `can_manage_torch_install`：False 时 UI 必须置灰「重装 PyTorch」整段
    - `manage_disabled_reason`：置灰原因（DCU 时为 :data:`DCU_REFUSE_REASON`）
    - `is_backend_unavailable`：装了 GPU build 但设备不可用（后端中立版）
    - `accelerator`：`AcceleratorInfo.as_dict()` 全量（卡名 / gfx arch / import_error）
    """
    torch_state = detect_torch()
    # 用 .get 而不是下标：detect_torch 的返回是公开 dict 契约，`backend` 是本次
    # 新加的 key，替身实现 / 老调用方可能只给原来那 5 个。缺失时按可管理处理，
    # 也就是保持 NVIDIA 既有行为不变。
    manageable = torch_state.get("backend") != "dcu"

    cuda_detect = onnxruntime_setup.detect_cuda()
    # DCU 上不问 recommend_cu_tag：它只懂 NVIDIA 驱动版本，会回 'cpu'，而 UI 若
    # 照着它渲染就变成推荐用户装 CPU wheel —— 在 DTK 镜像上是破坏性操作。
    # key 仍保留（前端 TorchStatus 类型要求非空），值走 'cpu' 且按钮已被置灰。
    recommended = recommend_cu_tag(cuda_detect.get("driver_version")) if manageable else "cpu"

    # 误装诊断：装了 CPU wheel 但有 NVIDIA GPU → UI 应该显著提示
    is_cpu_with_gpu = (
        manageable
        and torch_state["installed"]
        and torch_state["cuda_build"] == "cpu"
        and cuda_detect["available"]
    )
    # 装了 CUDA wheel 但 cuda.is_available()=False → 驱动 / WSL 问题，不是 pip 能修的
    is_cuda_build_unavailable = (
        manageable
        and torch_state["installed"]
        and torch_state["cuda_build"] not in (None, "cpu")
        and not torch_state["cuda_available"]
    )
    # 后端中立版：DCU 上容器没挂 /dev/kfd 或 DTK 装不全也走这条
    is_backend_unavailable = bool(
        torch_state["installed"]
        and torch_state["cuda_build"] not in (None, "cpu")
        and not torch_state["cuda_available"]
    )

    return {
        **torch_state,
        "cuda_detect": cuda_detect,
        "recommended_cu_tag": recommended,
        "is_cpu_with_gpu": is_cpu_with_gpu,
        "is_cuda_build_unavailable": is_cuda_build_unavailable,
        "is_backend_unavailable": is_backend_unavailable,
        "can_manage_torch_install": manageable,
        "manage_disabled_reason": None if manageable else DCU_REFUSE_REASON,
        "accelerator": accelerator.detect().as_dict(),
    }


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def _pip(args: list[str], timeout: int = 1800, stream: bool = False) -> tuple[int, str]:
    """跑 `<sys.executable> -m pip <args>`；返回 (rc, combined_output)。

    timeout 默认 30 分钟 —— torch + cuda 依赖打包后 ~3 GB，慢网下一小时也可能。
    stream=True：输出直接透传到终端（launch 场景用），不捕获，返回空字符串。
    stream=False（默认）：捕获并返回文本（API 端点 / 日志用）。
    """
    cmd = [sys.executable, "-m", "pip", *args]
    logger.info("[torch_setup] %s", " ".join(cmd))
    try:
        if stream:
            rc = subprocess.call(cmd, timeout=timeout)
            return rc, ""
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return 1, f"pip 超时（{timeout}s）: {exc}"
    except Exception as exc:  # noqa: BLE001
        return 1, f"pip 调用失败: {exc}"
    text = (out.stdout or "") + (out.stderr or "")
    return out.returncode, text


def _decide_target_tag(target: str) -> str:
    """auto / cu128 / cu126 / cu124 / cu118 / cpu → 实际 cu tag。

    'auto' → 用 nvidia-smi 推荐；其它直传。非法值抛 ValueError。
    DCU 上抛 RuntimeError —— 这里是 `/api/torch/reinstall` 写 pending marker **之前**
    的关卡，在这一步拒掉能保证 marker 根本不会落盘（不留一个下次启动就会破坏
    环境的定时炸弹）。
    """
    if not can_manage_torch_install():
        raise RuntimeError(DCU_REFUSE_REASON)
    if target == "auto":
        return recommend_cu_tag(onnxruntime_setup.detect_cuda().get("driver_version"))
    if target in SUPPORTED_INDEX_TAGS:
        return target
    raise ValueError(
        f"非法 target: {target!r}（应为 auto / {' / '.join(SUPPORTED_INDEX_TAGS)})"
    )


def _cleanup_zombie_dirs() -> list[str]:
    """清掉 site-packages 里 pip 失败时留下的 `~*` 僵尸目录。

    pip uninstall / install 失败后会留下 `~orch-...dist-info/`、`~orchvision/`
    一类的临时目录（前缀 `~` 是 pip 的占位符表示「正在重命名中」）。这些目录会
    导致下次 pip 装新 torch 时报 `Ignoring invalid distribution ~orch`，更严重
    的是让 `import torch` 看到残留 dist-info 但实际 .pyd 缺失。

    返回清理掉的路径列表，给日志用。Linux 上 site-packages 同样可能有 `~`-prefix
    残留（极少见但 pip 行为相同），所以不用 platform-skip。
    """
    site_packages = Path(sysconfig.get_path("purelib"))
    cleaned: list[str] = []
    if not site_packages.is_dir():
        return cleaned
    for entry in site_packages.glob("~*"):
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            cleaned.append(entry.name)
        except OSError as exc:
            logger.warning(
                "[torch_setup] removing the stale pip dir failed: path=%s err=%s",
                entry, exc,
            )
    if cleaned:
        logger.debug(
            "[torch_setup] removed stale pip dir: %s", ", ".join(cleaned),
        )
    return cleaned


def reinstall(target: str = "auto", stream: bool = False) -> dict[str, Any]:
    """卸装 torch + torchvision，按 target 重装。

    target: "auto" | "cu128" | "cu126" | "cu124" | "cu118" | "cpu"
    返回 `{"target", "tag", "index_url", "version", "stdout_tail",
            "restart_required": True, "cleaned_zombies": [...]}`。
    失败抛 RuntimeError。

    **重要**：torch 是 C extension，pip 卸装重装后**当前进程**已 import 的 .so/.pyd
    不会热替换。Server 进程 import 过 torch 时，pip uninstall 也会撞 [WinError 5]。
    所以本函数应在 launcher 进程跑（pending_install.apply_pending 调用），不在 server
    进程的 `/api/torch/reinstall` 端点里同步跑（那里只写 marker）。

    自愈：每次都先清 site-packages 里 `~*` 僵尸目录（之前失败留下的状态）。

    **DCU 上直接抛 RuntimeError**（见 :data:`DCU_REFUSE_REASON`）。这里的 guard 是
    最后一道：调用方（`/api/torch/reinstall` 的 `_decide_target_tag`、cli.py 的
    `--torch`）都已各自拒过一次，但 `pending_install` 的 marker 是文件，可能是
    换机器 / 手改 / 老版本留下的，落到 DCU 上执行就是不可逆破坏。
    """
    if not can_manage_torch_install():
        raise RuntimeError(DCU_REFUSE_REASON)

    tag = _decide_target_tag(target)
    index_url = _index_url_for(tag)

    # 第一步：清掉之前失败可能留下的僵尸目录。即使本次本来就没状态污染，也是
    # cheap operation（site-packages 列目录 + glob '~*'），代价微乎其微。
    cleaned = _cleanup_zombie_dirs()

    # 第二步：卸装 torch + torchvision（user-installed flash_attn / xformers 等不动，
    # 跟 torch ABI 强绑定，但卸装 torch 不会自动卸它们 —— 用户重启后再 enable / 重装）
    rc1, log1 = _pip(["uninstall", "-y", "torch", "torchvision"], stream=stream)

    # 卸装后再清一次僵尸目录（pip 卸装失败也可能留 `~`-prefix 残留）
    cleaned += _cleanup_zombie_dirs()

    # 第三步：安装。cu* 走 PyTorch 自家 index；cpu 也有自己的 index（不走 PyPI 默认避免歧义）
    install_args = ["install", "torch", "torchvision"]
    if index_url:
        install_args += ["--index-url", index_url]
    rc2, log2 = _pip(install_args, stream=stream)
    if rc2 != 0:
        raise RuntimeError(f"安装 torch ({tag}) 失败（rc={rc2}）:\n{log2}")

    stdout = log1 + log2
    tail = "\n".join(stdout.splitlines()[-40:])

    # dist-info 视角的新版本（进程里仍是旧 .pyd / .so）
    try:
        new_version = _pkg_version("torch")
    except PackageNotFoundError:
        new_version = None

    return {
        "target": target,
        "tag": tag,
        "index_url": index_url,
        "version": new_version,
        "stdout_tail": tail,
        "restart_required": True,
        "cleaned_zombies": cleaned,
    }
