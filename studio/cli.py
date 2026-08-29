"""跨平台启动器：替代 studio.bat 用 Python 管理前后端进程。

子命令：
    run    构建前端（如缺）+ 起后端（默认）
    dev    前后端开发模式（Vite 5173 + uvicorn 8765 --reload，并行）
    build  仅构建前端
    test   依次跑 pytest + vitest

入口：
    python -m studio                       # 等同 run
    python -m studio dev
    python -m studio build
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional

from studio.infrastructure.log_messages import msg

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "studio" / "web"
WEB_DIST = WEB_DIR / "dist"
NODE_MODULES = WEB_DIR / "node_modules"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def find_npm() -> Optional[str]:
    """Windows 优先 .cmd（CreateProcess 可直接跑），.ps1 兜底；Linux/Mac 走裸名。

    注意不要把裸 ``npm`` 放在 Windows 候选首位：Node.js 官方安装包在
    ``C:\\Program Files\\nodejs\\`` 同时铺了 ``npm`` (Git Bash 用的 bash 脚本)
    / ``npm.cmd`` / ``npm.ps1`` 三份，``shutil.which("npm")`` 在 Windows 上
    会先吃裸名那份，subprocess 直接报 WinError 193（不是有效 Win32 应用）。
    """
    candidates = ("npm.cmd", "npm.ps1", "npm") if os.name == "nt" else ("npm",)
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    return None


def find_python() -> str:
    """优先用当前解释器（venv 已激活则自然指对）。"""
    return sys.executable


_NPM_MIRROR = "https://mirrors.cloud.tencent.com/npm/"
_PIP_MIRROR = "https://mirrors.cloud.tencent.com/pypi/simple/"


_log = logging.getLogger("studio.cli")


def _say(line: str, level: str = "info") -> None:
    """统一 CLI 用户输出入口 —— `studio.cli` logger 的薄包装。

    行契约见 docs/design/logging-target-state.md §3.2 / §3.5：CLI 不再自己
    print，全部走 logger；`setup_logging("cli:<cmd>")` 装的 Human formatter 出
    `ts LEVEL studio.cli: msg` 前缀，级别由 `level` 真实表达（不再手写
    `[studio] ` / `警告：` 之类冒充级别的前缀）。终端可见级别由 ANIMA_LOG_LEVEL
    控（默认 INFO）。多行消息（安装提示 / 重装命令）作为一条记录，续行无前缀。

    level:
      - "info" / "success" → logger.info
      - "warning"          → logger.warning
      - "error"            → logger.error
    """
    text = str(line)
    if level == "error":
        _log.error(text)
    elif level == "warning":
        _log.warning(text)
    else:
        _log.info(text)


def _npm_argv(npm: str, args: list[str]) -> list[str]:
    """拼出真正可被 subprocess 执行的 argv。

    ``.ps1`` 无法被 ``CreateProcess`` 直接拉起，必须包 ``powershell.exe -File``；
    ``.cmd`` 和裸名（Linux）直接拼即可。
    """
    if npm.lower().endswith(".ps1"):
        return ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", npm, *args]
    return [npm, *args]


def _npm_call(npm: str, args: list[str], cwd: str, timeout: int = 180) -> int:
    """运行 npm 命令；超时则 kill 并返回 1。"""
    proc = subprocess.Popen(_npm_argv(npm, args), cwd=cwd)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return 1


def _frontend_package_files_changed_since_install() -> bool:
    marker = NODE_MODULES / ".package-lock.json"
    if not marker.exists():
        return False
    try:
        marker_mtime = marker.stat().st_mtime
        for f in (WEB_DIR / "package.json", WEB_DIR / "package-lock.json"):
            if f.exists() and f.stat().st_mtime > marker_mtime:
                return True
    except OSError:
        return False
    return False


def npm_install_if_missing(npm: str) -> int:
    _bin = "eslint.cmd" if os.name == "nt" else "eslint"
    deps_complete = NODE_MODULES.exists() and (NODE_MODULES / ".bin" / _bin).exists()
    package_files_changed = deps_complete and _frontend_package_files_changed_since_install()
    if deps_complete and not package_files_changed:
        return 0
    try:
        rel = NODE_MODULES.relative_to(REPO_ROOT)
    except ValueError:
        rel = NODE_MODULES
    if package_files_changed:
        _say(msg("cli.npm_install_stale"))
    else:
        _say(msg("cli.npm_install_missing", rel=rel))
    rc = _npm_call(npm, ["install"], str(WEB_DIR), timeout=180)
    if rc != 0:
        _say(
            f"npm install failed or timed out; retrying on the {_NPM_MIRROR} mirror",
            "warning",
        )
        rc = subprocess.call(
            _npm_argv(npm, ["install", "--registry", _NPM_MIRROR]),
            cwd=str(WEB_DIR),
        )
    return rc


def _pip_install(args: list[str]) -> int:
    """运行 pip install；失败时切换阿里云镜像重试。"""
    rc = subprocess.call([find_python(), "-m", "pip", "install"] + args)
    if rc != 0:
        _say(
            f"pip install failed; retrying on the {_PIP_MIRROR} mirror",
            "warning",
        )
        rc = subprocess.call(
            [find_python(), "-m", "pip", "install"] + args
            + ["-i", _PIP_MIRROR],
        )
    return rc


#: DCU 上要从 requirements 里剔掉的包名。DTK 版 torch / torchvision 由厂商镜像
#: 预装，PyPI 上没有对应 wheel —— 让 pip 看见 `torch>=2.0.0` 就会装 PyPI 的 CPU 版
#: 覆盖掉它们，环境不可逆报废。与 studio.sh 里同名的过滤保持一致（shell 首装路径
#: 走 shell 那份，本函数只管「已有 venv 缺包」的补装路径）。
_DCU_UNMANAGED_REQS: tuple[str, ...] = ("torch", "torchvision")


def _dcu_constraints() -> Optional[Path]:
    """DCU 上生成 pip 约束文件（把镜像预装的包钉住）；非 DCU / 无需约束返回 None。

    与 `_requirements_for_install` 的删行是两种机制，针对两类包：
    - 删行：torch / torchvision —— PyPI 上没有 DTK wheel，**不能装**。
    - 约束：transformers 等 —— 能装，但版本必须跟镜像一致（镜像里的 vllm /
      flash_attn 钉死了它们）。约束还能管住 requirements.txt 里没有的传递依赖
      （tokenizers 就是 transformers 拉进来的）。
    """
    from studio.services.runtime import torch as torch_setup  # noqa: PLC0415

    try:
        dest = torch_setup.write_dcu_constraints(REPO_ROOT / "tmp" / "constraints.dcu.txt")
    except OSError:
        # 写不出来不该阻断安装（约束是加固，不是必需）；但要让用户知道少了这层保护。
        _say(
            "海光 DCU：无法生成 pip 约束文件，本次安装不会把 transformers 等包钉在"
            "镜像版本上。若之后训练报 transformers / safetensors 相关错误，"
            "手动执行：pip install \"transformers==<镜像版本>\"",
            "warning",
        )
        return None
    if dest is not None:
        pinned = torch_setup.image_pinned_versions()
        _say(
            f"海光 DCU：已把镜像预装的 {len(pinned)} 个包钉住 —— "
            + ", ".join(f"{k}=={v}" for k, v in sorted(pinned.items()))
        )
    return dest


def _requirements_for_install() -> tuple[Path, Optional[Path]]:
    """返回 (要交给 pip 的 requirements 路径, 需要事后删的临时文件)。

    非 DCU 后端原样返回 `requirements.txt`（NVIDIA 路径逐字节不变）。DCU 上生成
    一份剔掉 torch / torchvision 的临时副本 —— 不改 requirements.txt 本体，因为它
    对 NVIDIA 用户仍然是必需约束，且 `check_requirements_changed.py` 的 hash
    marker 认的是原文件。
    """
    req = REPO_ROOT / "requirements.txt"
    from studio.services.runtime import torch as torch_setup  # noqa: PLC0415
    if torch_setup.can_manage_torch_install():
        return req, None

    kept: list[str] = []
    for line in req.read_text(encoding="utf-8").splitlines():
        # 只看行首的包名（`torch>=2.0.0` / `torchvision>=0.15.0`）。注意不能用
        # `startswith("torch")`：torchsde 是纯 Python 包、PyPI 上有、必须装。
        name = re.split(r"[<>=!~;\[\s]", line.strip(), maxsplit=1)[0].lower()
        if name in _DCU_UNMANAGED_REQS:
            continue
        kept.append(line)
    tmp = REPO_ROOT / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    filtered = tmp / "requirements.dcu.txt"
    filtered.write_text("\n".join(kept) + "\n", encoding="utf-8")
    _say(f"海光 DCU：requirements 里已剔除 {' / '.join(_DCU_UNMANAGED_REQS)}"
         "（DTK 版由镜像预装，pip 覆盖会报废环境）")
    return filtered, filtered


def _ensure_python_deps() -> int:
    """检查关键包（fastapi）是否安装，缺失时自动补装 requirements.txt。"""
    req = REPO_ROOT / "requirements.txt"
    if not req.exists():
        return 0
    try:
        import importlib.util
        if importlib.util.find_spec("fastapi") is not None:
            return 0
    except Exception:
        pass
    _say(msg("cli.reinstall_python_deps"))
    # DCU 上两件事一起做：
    # 1. 过滤 requirements（剔掉 torch / torchvision）—— PyPI 上没有 DTK wheel，
    #    装了会覆盖镜像预装的、环境不可逆报废。
    # 2. 加约束文件 —— transformers / safetensors / tokenizers / huggingface_hub
    #    钉在镜像预装的版本上。它们 PyPI 上有、装得上，但镜像里的 DCU 组件
    #    （vllm / flash_attn）钉死了版本，pip 装更新的到 venv 会遮蔽镜像版。
    target, cleanup = _requirements_for_install()
    constraints = _dcu_constraints()
    args = ["-r", str(target)]
    if constraints is not None:
        args += ["-c", str(constraints)]
    try:
        return _pip_install(args)
    finally:
        if cleanup is not None:
            cleanup.unlink(missing_ok=True)
        if constraints is not None:
            constraints.unlink(missing_ok=True)


def npm_build(npm: str) -> int:
    _say(msg("cli.build_frontend"))
    return subprocess.call(_npm_argv(npm, ["run", "build"]), cwd=str(WEB_DIR))


# ---------------------------------------------------------------------------
# 子进程协调
# ---------------------------------------------------------------------------


class ProcGroup:
    """同时管理多个子进程；任一进程退出或收到信号都把全部干掉。"""

    def __init__(self) -> None:
        self.procs: list[tuple[str, subprocess.Popen]] = []
        self._stopping = False

    def spawn(
        self,
        label: str,
        cmd: list[str],
        cwd: Optional[Path] = None,
    ) -> subprocess.Popen:
        creationflags = 0
        preexec_fn = None
        if os.name == "nt":
            # CREATE_NEW_PROCESS_GROUP 让我们能给整个组发 CTRL_BREAK_EVENT
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            # POSIX 下放进新进程组，杀的时候用 killpg
            preexec_fn = os.setsid  # type: ignore[assignment]
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            creationflags=creationflags,
            preexec_fn=preexec_fn,
        )
        _say(msg(
            "cli.process_spawned", label=label, pid=proc.pid,
            cmd=" ".join(cmd),
        ))
        self.procs.append((label, proc))
        return proc

    def wait_any(self) -> int:
        """阻塞到任一进程退出，返回该进程的 exit code。"""
        while True:
            for label, p in self.procs:
                rc = p.poll()
                if rc is not None:
                    if rc == 0:
                        _say(msg("cli.process_exited", label=label))
                    else:
                        _say(f"{label} exited with rc={rc}", "warning")
                    return rc
            try:
                # 让 KeyboardInterrupt 有机会触发
                threading.Event().wait(0.5)
            except KeyboardInterrupt:
                return 130

    def stop_all(self, grace: float = 10.0) -> None:
        if self._stopping:
            return
        self._stopping = True
        for label, p in self.procs:
            if p.poll() is not None:
                continue
            _say(msg("cli.stopping_process", label=label))
            try:
                if os.name == "nt":
                    p.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        for label, p in self.procs:
            try:
                p.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                _say(
                    f"{label} did not exit in {grace:.1f}s; "
                    "killing process tree", "warning",
                )
                p.kill()


# ---------------------------------------------------------------------------
# 命令实现
# ---------------------------------------------------------------------------


def _print_npm_install_hint() -> None:
    """`find_npm()` 返回 None 时输出平台相关安装提示。

    一条 error 记录（多行，续行无前缀）；root 环境去掉 sudo（直接 root 跑装包）。
    """
    lines = ["npm not found; install Node.js 18+ to build the frontend"]
    if os.name == "nt":
        lines.append(
            "  Windows: download the installer from https://nodejs.org, "
            "or run winget install OpenJS.NodeJS.LTS"
        )
    else:
        sudo = "" if (hasattr(os, "getuid") and os.getuid() == 0) else "sudo "
        lines.append(
            f"  Ubuntu/Debian: curl -fsSL https://deb.nodesource.com/setup_22.x "
            f"| {sudo}bash - && {sudo}apt-get install -y nodejs"
        )
        lines.append(
            "  or use nvm (no sudo): "
            "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh "
            "| bash && nvm install --lts"
        )
    lines.append("  re-run this command after installing")
    _say("\n".join(lines), "error")


def cmd_build(_args: argparse.Namespace) -> int:
    npm = find_npm()
    if not npm:
        _print_npm_install_hint()
        return 2
    rc = npm_install_if_missing(npm)
    if rc != 0:
        return rc
    rc = npm_build(npm)
    if rc == 0:
        _write_build_marker()
    return rc


def _current_git_head() -> Optional[str]:
    """当前仓库 HEAD commit hash；非 git 仓 / 没有 git 命令 → None。"""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _write_build_marker() -> None:
    """build 成功后把 HEAD 写到 dist/.built-from。云上下次启动直接对比 HEAD
    决定是否重建，绕开「git pull 不更新 mtime」的坑。"""
    head = _current_git_head()
    if not head:
        return
    try:
        (WEB_DIST / ".built-from").write_text(head, encoding="utf-8")
    except OSError:
        pass


def _spawn_browser_opener(url: str, *, delay: float = 1.0) -> None:
    """后台等服务起来后用默认浏览器打开 url；失败静默。"""

    def _wait_and_open() -> None:
        deadline = time.monotonic() + 30.0
        time.sleep(delay)
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1.5) as resp:
                    if 200 <= resp.status < 500:
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(0.5)
                continue
            except Exception:
                break
        try:
            webbrowser.open(url)
        except Exception:
            pass

    t = threading.Thread(target=_wait_and_open, name="studio-browser", daemon=True)
    t.start()


def _apply_pending_install() -> None:
    """启动期处理 server 进程不能完成的 pip 安装请求（torch 重装）。

    必须在 `_check_torch_cuda` 之前跑：那里会 import torch，之后 .pyd 被锁就装不动了。
    失败不抛 —— pending_install.apply_pending 内部已打印错误，让 launcher 继续起。
    """
    try:
        from studio.services.runtime import pending_install  # noqa: PLC0415
        pending_install.apply_pending()
    except Exception as exc:  # noqa: BLE001
        _say(
            f"handling the pending install request failed: {exc}; skipped",
            "warning",
        )


def _try_enable_flash_attn() -> None:
    """启动期检查 flash_attn 是否装好；装好就开 cosmos / anima 状态机。

    没装就 silently skip（_check_torch_cuda 不重复提示，flash_attn 是 nice-to-have）。
    动态 import 避免拖慢 cli import 时间（cosmos_predict2_modeling 加载触发 torch import）。
    """
    try:
        from studio.services.runtime import flash_attention as flash_attention_setup  # noqa: PLC0415
        if not flash_attention_setup.current_status()["installed"]:
            return
        from modeling.anima.cosmos_predict2_modeling import set_flash_attn_enabled  # noqa: PLC0415
        if set_flash_attn_enabled(True):
            _say(msg("cli.flash_attn_enabled"))
        else:
            # 装了 flash_attn 但 set_flash_attn_enabled 拒绝（_FLASH_ATTN_AVAILABLE=False）
            # 通常意味着 import 时挂了（CUDA 版本不匹配等）；不噪声只 warning 一行
            _say(
                "flash_attn is installed but importing the model layer failed; "
                "falling back to SDPA",
                "warning",
            )
    except Exception as exc:  # noqa: BLE001
        # Studio 启动不能为这一项加速 fail；记 warn 但放行
        _say(f"enabling flash_attn failed: {exc}; running without it", "warning")


def _apply_gpu_selection() -> None:
    """按 Settings 里的计算显卡选择注入 CUDA env（#491，多卡机器）。

    必须在 `_check_torch_cuda` import torch 之前——CUDA 只在 init 时读一次
    env；launcher 注入后 server 及其 spawn 的训练/出图子进程全部继承。
    launcher 是常驻进程（restart 循环），每轮重新调用让改动随重启生效。
    """
    try:
        from studio.services.runtime.gpu_select import apply_gpu_selection_env  # noqa: PLC0415

        apply_gpu_selection_env()
    except Exception:  # noqa: BLE001
        pass  # 选卡是 nice-to-have，不挡启动


def _check_torch_cuda() -> None:
    """启动期检查 torch 能不能上卡；CPU-only torch 跑训练 / 出图会极慢。

    按后端分派（后端识别一律问 `utils/accelerator.py`，见 docs/AGENTS.md）：

    - 加速器可用（NVIDIA 或 DCU）        → 一行 OK
    - NVIDIA：CPU-only build + 有 N 卡   → 大警告 + 重装命令（最常见误装）
    - NVIDIA：CPU-only build + 无 GPU    → 一行 info（用户确实在 CPU 机器上）
    - NVIDIA：CUDA build 但 cuda 不可用   → 警告（驱动 / WSL 问题）
    - DCU：DTK build 但设备不可用         → DCU 专属排错（设备节点 / DTK 安装）
    - CPU build 但机器上有 DCU 迹象       → 警告 venv 遮住了镜像预装的 DTK torch

    历史 bug：原实现直接用 `torch.version.cuda is None` 判「CPU-only wheel」，
    DTK wheel 的 `version.cuda` 恒为 None，于是海光机器上会被判成误装并被建议
    `pip install torch --index-url .../cu128` —— 那条命令会覆盖掉镜像预装的 DTK
    torch，环境不可逆报废。现在 CPU-only 的结论只在非 DCU 后端上下。
    """
    from utils.accelerator import VENDOR_LABEL, detect, probe_stdlib  # noqa: PLC0415

    # refresh=True：本函数在 cmd_run 的 restart loop 里每轮跑一次，而同一轮的
    # 前面刚可能发生过 `_ensure_python_deps`（补装 requirements）或
    # `_apply_pending_install`（pip 重装 torch）—— 那之前的探测结果（尤其
    # 「torch 没装」的 import_error）已经过期。启动期一次多余的探测代价可忽略，
    # 拿错结论却会让整段启动诊断静默。
    info = detect(refresh=True)
    if info.import_error:
        return  # torch 未装 / import 失败；_ensure_python_deps 会在更早路径处理

    if info.is_gpu:
        name = info.device_names[0] if info.device_names else "?"
        if info.backend == "dcu":
            # DTK 版本号是排错最关键的一条（用户报问题时先看它对不对得上镜像）。
            # 这条不走 msg()：DCU 专属，上游字典里没有对应 msg_id，硬塞会让
            # log_messages 与上游漂移。文案本身已是双语无关的版本号拼接。
            arch = f", {info.gcn_arch[0]}" if info.gcn_arch else ""
            _say(
                f"torch {info.torch_version} ({info.vendor_label} / HIP "
                f"{info.hip_version}, GPU: {name}{arch})"
            )
        else:
            _say(msg("cli.torch_gpu", version=info.torch_version, name=name))
        return

    if info.backend == "dcu":
        # DTK build 装着但设备用不了。**不给** pip 建议 —— DTK torch 是镜像预装的，
        # 重装只会让事情更糟；真正的原因几乎总在容器与驱动侧。
        _say(
            f"torch {info.torch_version} ({info.vendor_label} / HIP "
            f"{info.hip_version}) is installed but torch.cuda.is_available()=False; "
            f"training and generation would run on the CPU and be very slow. "
            f"Common causes:\n"
            f"  1. the container was started without the device nodes: needs "
            f"--device=/dev/kfd --device=/dev/dri (plus --group-add video; some "
            f"environments also need --security-opt seccomp=unconfined)\n"
            f"  2. the host DCU driver is missing or its version does not match the "
            f"DTK in the image (check whether hy-smi prints normally)\n"
            f"  3. the DTK runtime is incomplete (check that ROCM_PATH / "
            f"LD_LIBRARY_PATH point at DTK)\n"
            f"  diagnose with: python tools/probe_accelerator.py",
            "warning",
        )
        return

    if info.backend == "cpu":
        # CPU-only wheel。先按原有 NVIDIA 口径判误装（这条路径行为保持不变），
        # 再补一条 DCU 特有的误装形态：机器是海光的，但 venv 里的 torch 是 PyPI
        # CPU 版（venv 没开 --system-site-packages 就会遮住镜像预装的 DTK torch，
        # 或者曾经被 pip install -r requirements.txt 覆盖过）。
        try:
            from studio.services.runtime import onnxruntime as onnxruntime_setup  # noqa: PLC0415
            has_gpu = bool(onnxruntime_setup.detect_cuda().get("available"))
        except Exception:  # noqa: BLE001
            has_gpu = False
        if has_gpu:
            _say(
                f"NVIDIA GPU detected but the installed PyTorch is a CPU-only "
                f"build ({info.torch_version}); training and generation would run "
                f"on the CPU and be very slow (tens of seconds per step)\n"
                f"  pip uninstall torch torchvision -y\n"
                f"  # pick the index for your CUDA version, e.g. CUDA 12.8:\n"
                f"  pip install torch torchvision "
                f"--index-url https://download.pytorch.org/whl/cu128",
                "warning",
            )
            return
        if probe_stdlib().backend == "dcu":
            # DCU 专属：修法与 NVIDIA 完全不同（不能 pip 装 torch），所以单独一条。
            # 不走 msg()：上游字典没有这个 msg_id。
            _say(
                f"{VENDOR_LABEL['dcu']} hardware detected but the PyTorch in this "
                f"venv is a CPU build ({info.torch_version}); training and generation "
                f"would run on the CPU and be very slow. The DTK build of torch ships "
                f"preinstalled in the vendor image and CANNOT be installed from PyPI.\n"
                f"  likely cause: the venv was created without --system-site-packages, "
                f"which hides the image's DTK torch\n"
                f"  fix: delete venv/ and re-run ./studio.sh (it recreates the venv "
                f"with --system-site-packages), or use the image's python directly",
                "warning",
            )
            return
        _say(msg("cli.torch_cpu_only", version=info.torch_version))
        return

    # CUDA build 但运行时不可用：驱动 / WSL / 容器问题
    _say(
        f"torch {info.torch_version} is a CUDA {info.cuda_version} build but "
        f"torch.cuda.is_available()=False\n"
        f"  likely causes: the NVIDIA driver is not installed, the driver is "
        f"too old, or WSL is missing CUDA support",
        "warning",
    )


def _check_onnxruntime() -> None:
    """启动期 onnxruntime 状态检查（仅 detect，不装包）。

    对齐 xformers / flash-attention：未装时 silent skip（Tagging 页选 WD14 /
    CLTagger 会有徽章 + 引导按钮）。已装则打一行状态；CPU 包 + 有 GPU 走
    warn，提醒用户去 Settings 切 GPU 版。

    文案里的厂商名与 EP 名按后端取（`VENDOR_LABEL` / `onnx_gpu_provider`）：DCU 上
    GPU EP 是 MIGraphX 而不是 CUDA，且要装的是 DTK 配套的 onnxruntime（不是 PyPI
    的 onnxruntime-gpu），照抄「去 Settings 重装 GPU 版」会把用户引到装不上的路。
    """
    try:
        from studio.services.runtime import onnxruntime as onnxruntime_setup
        from utils.accelerator import VENDOR_LABEL, backend, onnx_gpu_provider  # noqa: PLC0415

        rt = onnxruntime_setup.current_runtime()
        if rt["installed"] is None:
            return

        installed = rt.get("installed") or "?"
        ver = rt.get("version") or "?"
        bk = backend()
        vendor = VENDOR_LABEL[bk]
        # `CUDAExecutionProvider` → `CUDA EP`、`MIGraphXExecutionProvider` →
        # `MIGraphX EP`。这么派生而不是写死映射，是为了 NVIDIA 那行文案与改动前
        # 逐字一致（"CUDA EP 可用"），同时新后端只要在 accelerator 里加一行即可。
        ep = (onnx_gpu_provider() or "GPU").replace("ExecutionProvider", " EP")
        if rt.get("cuda_available"):
            # ep 而非硬编码 "CUDA"：DCU 上 GPU EP 叫 MIGraphX / ROCm。
            _say(f"onnxruntime: {installed}=={ver} ({ep} available)")
            return

        if bk == "dcu":
            # DCU 侧没有 nvidia-smi，detect_cuda() 恒为 False，不能用它判「有没有卡」；
            # torch 侧的后端结论才是这台机器上唯一可信的 GPU 证据。
            _say(
                f"{vendor} detected but onnxruntime has no {ep} "
                f"(installed={installed}=={ver}); WD14 / CLTagger tagging will run on "
                f"the CPU (slower). GPU tagging needs the DTK-matched onnxruntime "
                f"(published on the Hygon channel, not PyPI's onnxruntime-gpu).",
                "warning",
            )
            return

        cuda = onnxruntime_setup.detect_cuda()
        if cuda.get("available"):
            _say(
                f"NVIDIA GPU detected but onnxruntime has only the CPU EP "
                f"(installed={installed}); WD14 and CLTagger tagging will run "
                f"on the CPU; reinstall the GPU build from "
                f"Settings → ONNX Runtime",
                "warning",
            )
        else:
            _say(msg("cli.onnxruntime_cpu", installed=installed, ver=ver))

    except Exception as exc:  # noqa: BLE001
        _say(f"onnxruntime status check failed: {exc}; ignored", "warning")


WEB_SRC = WEB_DIR / "src"


def _web_dist_is_stale() -> bool:
    """dist 是否落后于 src。两道检查并联，任一说 stale 就重建。

    1) git HEAD 比对：build 时把 HEAD 写到 dist/.built-from，启动时对比当前
       HEAD。云上 git pull 之后 HEAD 一定变，触发重建——这条是为了兜底
       "git pull 不更新文件 mtime" 在某些 git 版本下不可靠的坑。
    2) mtime 比对：dist/index.html 旧于 src/ 树或 package.json 等关键文件。
       这条是为了兜底本地未 commit 的修改——HEAD 不变但磁盘上的文件确实
       新过 dist，应当重建。

    曾经把 mtime 降级成 fallback（HEAD 一致就跳过），导致本地编辑后
    `studio run` 看不到变化。改并联后云上 git pull 行为不受影响，本地 dev
    iteration 也不需要每改完都 commit。
    """
    dist_index = WEB_DIST / "index.html"
    if not dist_index.exists():
        return True

    # 第一道：git HEAD 比对
    marker = WEB_DIST / ".built-from"
    head = _current_git_head()
    if head and marker.exists():
        try:
            built_from = marker.read_text(encoding="utf-8").strip()
            if built_from != head:
                return True
        except OSError:
            pass

    # 第二道：mtime 比对
    try:
        dist_mtime = dist_index.stat().st_mtime
        src_latest = max(
            (p.stat().st_mtime for p in WEB_SRC.rglob("*") if p.is_file()),
            default=0.0,
        )
        # package.json / vite.config 改了也算
        for f in (WEB_DIR / "package.json", WEB_DIR / "vite.config.ts", WEB_DIR / "tsconfig.json"):
            if f.exists():
                src_latest = max(src_latest, f.stat().st_mtime)
        if src_latest > dist_mtime:
            return True
    except OSError:
        pass

    return False


_RESTART_FLAG = REPO_ROOT / "tmp" / "restart"

# PR-D — installer 自检（ADR 0002）。cmd_run 入口快照这三个文件的 sha256；
# 每次 server 退出 + 收到 restart 请求时再算一次，任一变化 → 返回退出码 42
# 让 wrapper（studio.sh / studio.bat）整体 exec 自己。原因：
#
# - cli.py 本身变更 → 旧 python 进程加载的是旧 cli.py，next-iteration 的 inner
#   loop 仍走老逻辑；只有让 wrapper 重新拉 `python -m studio` 才能拿到新 cli.py。
# - studio.sh / studio.bat 变更 → bash 已把 loop 体加载进内存，cmd.exe 也可能
#   缓存 .bat 解析结果；必须让 shell 进程 exec 自己拿到新 wrapper。
#
# 三个文件中任一变化都走同一协议（最简单 / 最稳）。
_INSTALLER_FILES: tuple[Path, ...] = (
    REPO_ROOT / "studio" / "cli.py",
    REPO_ROOT / "studio.sh",
    REPO_ROOT / "studio.bat",
)
_INSTALLER_RELOAD_EXIT_CODE = 42


def _installer_hashes() -> dict[str, Optional[str]]:
    """快照 installer 文件 sha256。文件不存在 → 值为 None（跨平台：Linux 上
    studio.bat 不存在，Windows 上 studio.sh 不存在；存在性也算入比对）。"""
    result: dict[str, Optional[str]] = {}
    for p in _INSTALLER_FILES:
        try:
            result[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            result[p.name] = None
    return result


def _apply_update_pending() -> None:
    """启动期处理 webui 触发的 update 请求（ADR 0002 / PR-B）。

    server 端 POST /api/system/update 写 studio_data/.update_pending 后通过
    SIGINT 退出。我们在 cli.py 主循环里下一轮 bootstrap 之前调这个函数完成：

    1. git fetch + git reset --hard {target}
    2. requirements.txt 变了 → pip install -r（增量）
    3. studio/web/package.json 变了 → npm install
    4. 清 update cache 让下次 check_update 重 fetch

    失败不抛 —— `apply_pending` 内部把错误写到 studio_data/.update_log，
    让 cli.py 继续走后面的 bootstrap 把 server 至少起回来（用户能在 UI
    看到"上次 update 失败"提示）。
    """
    try:
        from studio.services.runtime import updater  # noqa: PLC0415
        from studio.infrastructure.task_log import TaskLog  # noqa: PLC0415
        if not updater.has_pending():
            return
        # 与 _say 同一个 logger，updater 内部升级到级别方法时黄/红行自动落对
        updater.apply_pending(emit=TaskLog(_log))
    except Exception as exc:  # noqa: BLE001
        _say(f"applying the pending update failed: {exc}; skipped", "warning")


def _maybe_force_torch(args: argparse.Namespace) -> int:
    """--torch <tag> 指定时，检查当前安装是否匹配；不匹配则立即重装（流式输出）。
    仅在 launcher 启动期调一次，重装完由 restart 机制加载新 torch。

    海光 DCU 上直接拒绝（返回 1 让启动停下）：用户显式传了这个 flag，静默忽略会
    让他以为换成功了；而真去执行会覆盖掉镜像预装的 DTK torch，环境不可逆报废。
    """
    tag = getattr(args, 'torch', None)
    if not tag:
        return 0
    from studio.services.runtime import torch as torch_setup  # noqa: PLC0415
    # 在 detect_torch() 之前问：can_manage_torch_install() 不会 import torch，
    # 这样 DCU 分支从头到尾不产生 torch import 副作用（跟 pending_install 的
    # 「pip 之前不许 import torch」约定同源）。
    if not torch_setup.can_manage_torch_install():
        _say(f"--torch {tag} 被拒绝：{torch_setup.DCU_REFUSE_REASON}", "error")
        return 1
    current = torch_setup.detect_torch()
    current_build = current.get('cuda_build') or ('未安装' if not current.get('installed') else 'unknown')
    if current.get('installed') and current.get('cuda_build') == tag:
        _say(msg("cli.torch_already_tag", tag=tag))
        return 0
    _say(msg(
        "cli.torch_reinstall_start", tag=tag, current_build=current_build,
    ))
    try:
        res = torch_setup.reinstall(tag, stream=True)
        _say(msg(
            "cli.torch_reinstall_done",
            version=res.get("version"), tag=res.get("tag"),
        ))
        return 0
    except KeyboardInterrupt:
        _say("torch reinstall interrupted by user; skipped", "warning")
        return 0
    except RuntimeError as exc:
        _say(f"torch reinstall failed: {exc}", "error")
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    """`run` 主循环。

    内层 loop：每次 server 退出后检查 `tmp/restart` 标志（由 server 端
    `/api/system/restart` 写）。存在则删除标志 + 重走 bootstrap + 重起 server；
    不存在则跳出，正常退出。

    重启协议详见 `docs/adr/0002-webui-self-update.md`。外层 shell wrapper
    (`studio.sh` / `studio.bat`) 也有同样的 loop 兜底（cli.py 异常退出但
    flag 还在的场景），并且响应退出码 42 把自己 exec 一遍（PR-D installer
    自检：当 cli.py / studio.sh / studio.bat 本身被 update 修改后，需要从
    磁盘重新加载 wrapper + Python 解释器）。

    冷启动只打开一次浏览器；重启时复用已存在的 webui 标签页（前端轮询
    `/api/health` 自动 reconnect），不重复弹新窗口。
    """
    opened_browser = False
    # --torch 强制重装（仅首次，不在 restart 循环里重复）
    rc = _maybe_force_torch(args)
    if rc != 0:
        return rc
    # PR-D：快照启动期 installer 文件 sha256；server 退出后重算，变化则
    # 退出码 42 让 wrapper 整体 exec 自己。
    startup_installer = _installer_hashes()
    while True:
        rc = _ensure_python_deps()
        if rc != 0:
            return rc

        # 检测 update pending（ADR 0002 / PR-B）：上一轮 server 写了
        # studio_data/.update_pending 并请求重启，这里在重新 bootstrap 之前
        # 先 git pull + 必要时 pip install / npm install，确保后续的 stale
        # 检查 / native module 加载用的都是新版代码 / 新版依赖。
        _apply_update_pending()

        if not args.no_build:
            if not WEB_DIST.exists():
                _say(msg("cli.frontend_dist_missing"))
                rc = cmd_build(args)
                if rc != 0:
                    return rc
            elif _web_dist_is_stale():
                _say(msg("cli.frontend_dist_stale"))
                rc = cmd_build(args)
                if rc != 0:
                    return rc
        if not getattr(args, 'skip_pending', False):
            _apply_pending_install()
        _apply_gpu_selection()
        _check_torch_cuda()
        _try_enable_flash_attn()
        _check_onnxruntime()
        url = f"http://{args.host}:{args.port}/"
        _say(msg("cli.backend_started", url=url))
        if not args.no_browser and not opened_browser:
            _spawn_browser_opener(url)
            opened_browser = True
        try:
            rc = subprocess.call(
                [find_python(), "-m", "studio.server", "--host", args.host, "--port", str(args.port)]
            )
        except KeyboardInterrupt:
            # 终端 Ctrl+C：CTRL_C_EVENT 同时广播给 server 子进程（它自己走
            # graceful shutdown）；父进程这边阻塞在 wait 的 KeyboardInterrupt
            # 要等子进程退干净后才抛出来。用户主动停机，不打 traceback。
            _say(msg("cli.stopped_ctrl_c"))
            return 130

        if not _RESTART_FLAG.exists():
            return rc

        # PR-D：installer 自检。restart flag 存在的前提下，若 cli.py /
        # studio.sh / studio.bat 任一变化，**保留** flag 并返回 42，让 wrapper
        # 走 exec self 路径。flag 保留是关键 —— wrapper 检测到 (exit==42 &&
        # flag exists) 才会 re-exec；只剩 flag 而 exit!=42 则走普通 restart。
        if _installer_hashes() != startup_installer:
            _say(msg("cli.launcher_reload"))
            return _INSTALLER_RELOAD_EXIT_CODE

        # 收到重启请求：删除标志 + loop 回去重新 bootstrap
        try:
            _RESTART_FLAG.unlink()
        except OSError:
            pass
        _say(msg("cli.restart_requested"))


def cmd_dev(args: argparse.Namespace) -> int:
    rc = _maybe_force_torch(args)
    if rc != 0:
        return rc
    rc = _ensure_python_deps()
    if rc != 0:
        return rc
    npm = find_npm()
    if not npm:
        _print_npm_install_hint()
        return 2
    rc = npm_install_if_missing(npm)
    if rc != 0:
        return rc
    if not getattr(args, 'skip_pending', False):
        _apply_pending_install()
    _apply_gpu_selection()
    _check_torch_cuda()
    _try_enable_flash_attn()
    _check_onnxruntime()

    pg = ProcGroup()
    try:
        pg.spawn("frontend", _npm_argv(npm, ["run", "dev", "--", "--port", str(args.fe_port)]), cwd=WEB_DIR)
        pg.spawn(
            "backend",
            [
                find_python(),
                "-m",
                "studio.server",
                "--host", args.host,
                "--port", str(args.port),
                "--reload",
            ],
        )
        frontend_url = f"http://127.0.0.1:{args.fe_port}/"
        _say(msg(
            "cli.dev_urls", frontend_url=frontend_url,
            backend_url=f"http://{args.host}:{args.port}/",
        ))
        if not args.no_browser:
            # dev 模式打开 Vite 端口（HMR 能用），不开 backend 端口
            _spawn_browser_opener(frontend_url, delay=2.0)
        rc = pg.wait_any()
    finally:
        pg.stop_all()
    return rc


def cmd_test(_args: argparse.Namespace) -> int:
    """跑 pytest + vitest。任一失败 → 非零退出。"""
    _say(msg("cli.pytest_start"))
    rc = subprocess.call([find_python(), "-m", "pytest", "tests/"], cwd=str(REPO_ROOT))
    if rc != 0:
        return rc
    npm = find_npm()
    if not npm:
        _say(msg("cli.vitest_skipped_no_npm"))
        return 0
    if not NODE_MODULES.exists():
        _say(msg("cli.vitest_skipped_no_modules"))
        return 0
    _say(msg("cli.vitest_start"))
    return subprocess.call(_npm_argv(npm, ["run", "test"]), cwd=str(WEB_DIR))


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="studio", description="AnimaStudio 启动器")
    sub = p.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="构建前端（如缺）+ 起后端")
    p_run.add_argument("--host", default="127.0.0.1")
    p_run.add_argument("--port", type=int, default=8765)
    p_run.add_argument("--no-build", action="store_true",
                       help="即使 dist 不存在也不自动 build")
    p_run.add_argument("--no-browser", action="store_true",
                       help="启动后不自动打开浏览器")
    p_run.add_argument("--skip-pending", action="store_true",
                       help="跳过 pending pip 安装（torch 重装等），直接启动")
    p_run.add_argument("--torch", metavar="TAG",
                       help="强制指定 torch CUDA 版本（cu128/cu126/cu124/cu118/cpu），"
                            "与当前不符时自动重装。CPU 租赁机预装 GPU torch 时使用。"
                            "海光 DCU 上会被拒绝（DTK torch 由镜像预装）。")
    p_run.set_defaults(func=cmd_run)

    p_dev = sub.add_parser("dev", help="前后端开发模式")
    p_dev.add_argument("--host", default="127.0.0.1")
    p_dev.add_argument("--port", type=int, default=8765,
                       help="后端 uvicorn 端口（默认 8765）")
    p_dev.add_argument("--fe-port", type=int, default=5173,
                       help="前端 Vite 开发服务器端口（默认 5173）")
    p_dev.add_argument("--no-browser", action="store_true",
                       help="启动后不自动打开浏览器")
    p_dev.add_argument("--skip-pending", action="store_true",
                       help="跳过 pending pip 安装（torch 重装等），直接启动")
    p_dev.add_argument("--torch", metavar="TAG",
                       help="强制指定 torch CUDA 版本（cu128/cu126/cu124/cu118/cpu）。"
                            "海光 DCU 上会被拒绝（DTK torch 由镜像预装）。")
    p_dev.set_defaults(func=cmd_dev)

    p_build = sub.add_parser("build", help="仅构建前端")
    p_build.set_defaults(func=cmd_build)

    p_test = sub.add_parser("test", help="跑 pytest + vitest")
    p_test.set_defaults(func=cmd_test)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args_list = list(argv) if argv is not None else sys.argv[1:]
    # 没有子命令时默认 run（如 studio.sh --port 6006 → run --port 6006）。
    # 找第一个不以 '-' 开头的参数，判断是否是已知子命令；不是则插入 run。
    _subcmds = {'run', 'dev', 'build', 'test'}
    _first_pos = next((a for a in args_list if not a.startswith('-')), None)
    if _first_pos not in _subcmds:
        args_list = ['run'] + args_list
    args = parser.parse_args(args_list)
    # 统一日志体系 (ADR-0009 / docs/design/logging-target-state.md §3.5)。
    # file=False — CLI 是 5s 短命周期，启动信息不进 studio.log。console=True 让
    # `_say` → studio.cli logger 走人读 stderr（Human 格式，级别读 ANIMA_LOG_LEVEL）。
    # 必须在任何 `_say` 之前装好：所有 `_say` 都在 args.func 内，parse_args 之前
    # 只有 argparse 自己的 usage/help 输出（不算日志）。env
    # ANIMA_LOGGING_NO_BOOTSTRAP=1 时 noop（测试态）。
    from .infrastructure.logging import setup_logging
    setup_logging(f"cli:{args.cmd}", file=False, console=True)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
