#!/usr/bin/env python
"""Bootstrap helper: 按检测到的 NVIDIA 驱动版本输出 PyTorch wheel index URL。

studio.bat / studio.sh 在 venv **首装**时调本脚本，先按 GPU 装对的 torch，再装
requirements.txt。约束 `torch>=2.0.0` 已被首步满足，pip 不会再覆盖成 PyPI 默认 CPU 版。

Stdlib only —— 在 venv 刚装好（只有 pip + setuptools）时也能跑。

输出（默认模式）：
- 检测到合适驱动 → stdout 一行 URL，如 `https://download.pytorch.org/whl/cu128`
- 没装 nvidia-smi / 解析失败 / 驱动太旧 → 静默无输出（caller 用 PyPI 默认）
- **海光 DCU** → 静默无输出（caller 必须完全不碰 torch，见下）
- 永远 exit 0，不让 bootstrap 因这一项 fail

`--backend` 模式：输出后端标识（`cuda` / `dcu` / `cpu`），供 shell 分支用。
默认模式在 DCU 与「无驱动」两种情况下都静默，而这两种情况 caller 的动作**相反**
（前者跳过 torch，后者装 PyPI 默认），所以必须有第二个问法把它们区分开。

**为什么 DCU 必须静默**：海光 DTK 的 torch wheel 由厂商镜像预装，不在 PyPI 上，
且与镜像内 DTK 运行时严格配套。任何 `pip install torch` 都会把它替换成 PyPI 的
CPU 版，环境直接报废且无法用 pip 装回来（用户只能重建容器）。见
`utils/accelerator.should_manage_torch_install`。

驱动→cu wheel 映射：与 studio/services/runtime/torch.py:_DRIVER_TO_BEST_CU 同步。
单独 duplicate 是为了 bootstrap 阶段不依赖 studio.services 子模块加载链。
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# 后端识别走 utils/accelerator.py（全仓库单一权威源，见 docs/AGENTS.md §3.1）。
# 那个模块本身也是纯 stdlib（torch 只是可选 import，probe_stdlib() 完全不碰 torch），
# 所以在「venv 里只有 pip」的 bootstrap 阶段照样能加载。
#
# 显式插 repo root 到 sys.path 而不是依赖 CWD：本脚本约定从仓库根跑
# （`python tools/select_torch_index.py`），此时 sys.path[0] 是 `tools/`，`utils`
# 不在搜索路径上。用 __file__ 推 repo root 对「从别处调用绝对路径」也成立。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 注：维护此表时同步更新 studio/services/torch_setup.py:_DRIVER_TO_BEST_CU
_DRIVER_TO_CU: list[tuple[int, str]] = [
    (555, "cu128"),
    (550, "cu126"),
    (545, "cu124"),
    (470, "cu118"),
]
_PYPI_BASE = "https://download.pytorch.org/whl"


def detect_driver_major() -> int | None:
    """跑 nvidia-smi 拿驱动版本主号；失败 / 不存在返回 None。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    line = (out.stdout or "").strip().split("\n")[0]
    m = re.match(r"^(\d+)\.", line)
    if not m:
        return None
    return int(m.group(1))


def select_index_url(driver_major: int | None) -> str | None:
    """driver 主号 → PyTorch wheel index URL；驱动太旧 / None → None。"""
    if driver_major is None:
        return None
    for threshold, tag in _DRIVER_TO_CU:
        if driver_major >= threshold:
            return f"{_PYPI_BASE}/{tag}"
    return None


def detect_backend() -> str:
    """当前机器的加速器后端：`cuda` / `dcu` / `cpu`。

    import 失败（仓库文件缺失 / 被单文件拷出来跑）时回落 `cuda` —— 那是本脚本
    原有的行为口径（继续走 nvidia-smi 探测，探不到就静默），不会把 NVIDIA 机器
    的既有路径改坏。代价是这种畸形环境下 DCU 保护失效，但保护本来也依赖仓库
    文件在位。
    """
    try:
        from utils.accelerator import probe_stdlib  # noqa: PLC0415
    except Exception:  # noqa: BLE001  bootstrap 阶段任何 import 异常都不能让它崩
        return "cuda"
    return probe_stdlib().backend


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    backend = detect_backend()

    if "--backend" in args:
        # 不带换行 —— 与 URL 输出同口径
        sys.stdout.write(backend)
        return 0

    if backend == "dcu":
        # 静默 = caller 不装 torch。**不要**在这里输出任何 fallback index：
        # DTK torch 是镜像预装的，PyPI 上没有对应 wheel，装什么都是覆盖破坏。
        return 0

    url = select_index_url(detect_driver_major())
    if url:
        # 不带换行 —— shell `for /f` / `$()` 处理时少踩坑
        sys.stdout.write(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
