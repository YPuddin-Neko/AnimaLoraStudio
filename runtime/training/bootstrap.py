"""依赖检测、YAML 配置加载、进度条初始化等启动期工具。

抽自原 runtime/anima_train.py L60-180（ADR 0003 PR-A）。

公开函数：
- ensure_dependencies — 检测并可选自动安装缺失依赖
- load_yaml_config / apply_yaml_config — YAML 配置 → args 合并
- init_progress — Rich 进度条初始化
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from studio.infrastructure.log_messages import msg

logger = logging.getLogger(__name__)


#: torch 生态包：``--auto-install`` 在海光 DCU 上**不能**碰这些。
#: DTK 的 torch / torchvision 由厂商镜像预装，wheel 不在 PyPI 上。一旦 pip 从 PyPI
#: 装 torchvision，它会连带拉一个版本匹配的 **CPU 版 torch** 覆盖掉预装的 DTK torch
#: —— 环境当场报废且无法用 pip 装回来，用户只能重建容器。宁可 fail-fast 让用户自己
#: 从 DTK 渠道补，也不能自动装。
_TORCH_FAMILY_PACKAGES = frozenset({"torch", "torchvision", "torchaudio"})


def ensure_dependencies(auto_install: bool = False) -> None:
    """检测并可选自动安装缺失依赖。"""
    required = {
        "numpy": "numpy",
        "PIL": "Pillow",
        "safetensors": "safetensors",
        "transformers": "transformers",
        "einops": "einops",
        "torchvision": "torchvision",
        "yaml": "pyyaml",
    }
    missing = []
    for module_name, pip_name in required.items():
        try:
            __import__(module_name)
        except Exception:
            missing.append(pip_name)
    if not missing:
        return
    missing_list = ", ".join(sorted(set(missing)))
    if not auto_install:
        logger.error(
            "Dependency check failed: missing=%s — training aborted; "
            "install with: %s -m pip install %s",
            missing_list, sys.executable, missing_list,
        )
        raise SystemExit(1)
    # DCU 上把 torch 生态包从自动安装清单里剔掉（理由见 _TORCH_FAMILY_PACKAGES）。
    # 只在 DCU 上 gate：NVIDIA 路径行为保持原样，避免为了这个护栏改动既有用户的体验。
    auto_targets = sorted(set(missing))
    from utils.accelerator import is_dcu

    if is_dcu():
        blocked = [p for p in auto_targets if p in _TORCH_FAMILY_PACKAGES]
        if blocked:
            # 这条走 logger.error 而不是 print：日志改写后 run.log 是单一出口，
            # print 会绕过级别与格式（设计 D3）。而它紧接着 SystemExit(1)，
            # 属于致命错误而非提示。
            logger.error(
                "Refusing to auto-install on Hygon DCU: %s\n"
                "  DTK torch/torchvision are preinstalled in the vendor image and are\n"
                "  NOT on PyPI. Installing from PyPI would pull a CPU-only torch and\n"
                "  overwrite the DTK build, breaking the environment beyond pip repair.\n"
                "  Get the matching DTK wheels from the Hygon developer channel instead.",
                ", ".join(blocked),
            )
            raise SystemExit(1)

    logger.info(msg("train.deps_installing", missing=missing_list))
    cmd = [sys.executable, "-m", "pip", "install", *auto_targets]
    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:
        logger.exception(
            "Dependency auto-install failed: %s — training aborted; "
            "install manually: %s -m pip install %s",
            exc, sys.executable, missing_list,
        )
        raise SystemExit(1)
    still_missing = []
    for module_name, pip_name in required.items():
        try:
            __import__(module_name)
        except Exception:
            still_missing.append(pip_name)
    if still_missing:
        still_list = ", ".join(sorted(set(still_missing)))
        logger.error(
            "Dependency check failed after auto-install: missing=%s — training "
            "aborted; install manually: %s -m pip install %s",
            still_list, sys.executable, still_list,
        )
        raise SystemExit(1)


def load_yaml_config(config_path):
    """加载 YAML 配置文件。"""
    try:
        import yaml
    except ImportError:
        logger.error(
            "Dependency check failed: missing=pyyaml — config file cannot be read, "
            "training aborted; install with: %s -m pip install pyyaml",
            sys.executable,
        )
        raise SystemExit(1)

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def apply_yaml_config(args, config):
    """将 YAML 与 CLI 显式参数合并，经 TrainingConfig 完整构造后返回新 args。

    config 管线刀 1（R1，docs/design/config-pipeline-refactor.md）：trainer 与
    Studio 走同一条 pydantic 加载路径 —— 字段迁移 / FAMILY_CONFIG_DEFAULTS
    族默认 overlay / 互斥与能力校验全部单点生效，本函数不再手工重放迁移
    （旧 merge_yaml_into_namespace 绕过 validator 年代的产物）。

    命令行显式参数优先于 YAML：parse_args 以 suppress_defaults 构建 parser，
    args 只含显式键，优先级是精确判定而非「值==默认值」近似。

    校验失败以一条 ERROR 记录（逐条错误作续行）落日志后 SystemExit(2) ——
    与能力防线同款 fail-fast，supervisor 从 run.log 尾部取错误块作为任务错误信息。
    """
    from pydantic import ValidationError

    from studio.infrastructure.argparse_bridge import namespace_from_config
    from studio.schema import TrainingConfig

    try:
        return namespace_from_config(args, dict(config or {}), TrainingConfig)
    except ValidationError as exc:
        errors = exc.errors()
        lines = [f"Config validation failed: {len(errors)} problem(s) — training aborted"]
        for err in errors:
            loc = ".".join(str(p) for p in err["loc"]) or "config"
            lines.append(f"  {loc}: {err['msg']}")
        logger.error("\n".join(lines))
        raise SystemExit(2) from exc


def init_progress(show_progress, total_steps):
    """初始化 Rich 进度条。

    返回 `(progress, task_id, kind)`：
    - 关闭进度时返回 `(None, None, None)`
    - Rich 可用时返回 `(Progress 实例, task_id, "rich")`
    - Rich 缺失时返回 `("plain", None, None)`（main() 据此走纯文本进度）

    非 tty（studio spawn 的 pipe）强制降级走 log_every 纯文本分支——
    rich 在 pipe 下既刷屏又吃掉 step 行（log_every 是 elif），存量
    config 固化的 ``no_progress: false``（老默认 + 字段现已 hidden）
    曾让 task log 里一行 step 日志都没有。裸终端 CLI 不受影响。
    """
    if not show_progress:
        return None, None, None
    import sys as _sys

    if not (hasattr(_sys.stdout, "isatty") and _sys.stdout.isatty()):
        return None, None, None
    try:
        from rich.progress import (
            BarColumn, MofNCompleteColumn, Progress, TextColumn,
            TimeElapsedColumn, TimeRemainingColumn,
        )
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("loss={task.fields[loss]:.4f}"),
            TextColumn("lr={task.fields[lr]:.2e}"),
            TextColumn("speed={task.fields[speed]:.2f} it/s"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=10,
        )
        task = progress.add_task("train", total=total_steps, loss=0.0, lr=0.0, speed=0.0)
        return progress, task, "rich"
    except Exception:
        return "plain", None, None
