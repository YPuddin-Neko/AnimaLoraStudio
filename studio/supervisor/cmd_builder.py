"""默认 cmd builder + worker EVENT 协议常量（PR-4 从 supervisor.py 抽出）。

`Supervisor.__init__` 接受 `cmd_builder` / `job_cmd_builder` 注入参数，方便
测试替换；这里实现 supervisor 内默认走真实 runtime/anima_train.py / workers
模块的版本。

多卡（DDP）启动也在这里决定：`utils/distributed.py` 只**读** torchrun 注入的
`RANK` / `LOCAL_RANK` / `WORLD_SIZE`，自己不拉进程，所以「起几个进程」这件事
的唯一决策点是本模块（见 ADR 0016 双后端说明与 utils/distributed 模块 docstring）。
"""
from __future__ import annotations

import logging
import socket
import sys
from pathlib import Path
from typing import Any, Callable

from ..paths import REPO_ROOT, task_dir, task_monitor_state_path

# R-1 起调度准入不再用这个集合（改走 resources.py 的档位模型：exclusive /
# light / io，见 docs/design/queue-resource-model-0.17.md）。此处保留为
# 「吃 GPU 的 job kind」派生事实（= 非 io 档），供文档性断言使用。
from .resources import JOB_KIND_RESOURCE_CLASS as _JOB_CLASS, RESOURCE_IO as _IO

GPU_BOUND_JOB_KINDS: frozenset[str] = frozenset(
    k for k, c in _JOB_CLASS.items() if c != _IO
)

logger = logging.getLogger(__name__)

#: TrainingConfig 里「起几个训练进程」的字段名（单一权威源在
#: studio/domain/training.py:TrainingConfig.ddp_num_processes）。这里只按名字
#: 读 yaml、不 import TrainingConfig —— supervisor 的 dispatch 热路径不该为了
#: 取一个整数就把整套 pydantic schema + 族能力矩阵拉进来。
DDP_FIELD = "ddp_num_processes"

# Worker → supervisor 的结构化事件标记。worker 写
#   __EVENT__:my_event_type:{"foo":1,"bar":"x"}
# 到 stdout，supervisor 在 _on_line 里识别并 publish 成 typed SSE 事件
# （job_id / project_id 自动注入），不会进 job_log。比专门搭 IPC 轻。
_EVENT_MARKER = "__EVENT__:"


EventCallback = Callable[[dict[str, Any]], None]
CmdBuilder = Callable[[dict[str, Any], Path], list[str]]
JobCmdBuilder = Callable[[dict[str, Any]], list[str]]


def _read_ddp_num_processes(config_path: Path) -> int:
    """从 task 的 config.yaml 读进程数；任何异常一律降级回 1（单卡）。

    读盘失败 / yaml 损坏 / 值非法时**绝不抛**：本函数跑在 supervisor 的
    `_tick` 里，而 `_tick` 的异常被 `_loop` catch + log 后按 poll 间隔无限重试
    （task 停在 pending，用户在 UI 上只看到「排队中」永远不动、日志被刷爆）。
    降级回单卡至少让训练能跑起来，配置真的写错了由训练侧
    `utils.distributed.init()` 的 LOCAL_RANK 越界检查报出可读原因。
    """
    try:
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        n = int(raw.get(DDP_FIELD, 1) or 1)
        return n if n > 1 else 1
    except Exception:  # noqa: BLE001  见 docstring：这里不能抛
        logger.exception("读取 %s 失败，按单卡处理: %s", DDP_FIELD, config_path)
        return 1


def _free_master_port() -> int:
    """让内核挑一个空闲端口给 torchrun 的 rendezvous。

    不用 torchrun 默认的 29500：同机上可能有第二个 Studio 实例、别人手跑的
    torchrun、或上一次训练崩溃后残留在 TIME_WAIT 的 socket，端口被占则第二个
    任务直接起不来（报 `Address already in use`，与训练配置无关，用户无从下手）。

    也不用 `--standalone`：它的端口策略在 torch 版本间变过（早期固定 29400，
    后来才改成 port 0 动态选），而 requirements 只钉 `torch>=2.0.0`；显式
    `--master_port` 走 static rendezvous 在整个 2.x 上行为一致。

    bind(0) 拿到端口后立刻关闭，到 torchrun 真正 bind 之间有毫秒级 TOCTOU
    窗口理论上会被别人抢走。相比「固定端口必然撞」这是明确的改善，不值得为
    这个概率再套一层重试。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _torchrun_prefix(nproc: int, task_id: Any) -> list[str]:
    """多卡时插在 `python` 与训练脚本之间的 torchrun 前缀；单卡返回 `[]`。

    为什么用 ``python -m torch.distributed.run`` 而不是裸 ``torchrun``
    ----------------------------------------------------------------
    ``torchrun`` 是 pip 装 torch 时生成的 console script，落在 venv 的
    ``bin/`` / ``Scripts/``，能不能找到取决于子进程的 PATH。supervisor 用
    ``sys.executable`` 起子进程但**不动 PATH**（见 `_popen`），Studio 从桌面
    快捷方式 / 服务方式启动时 PATH 里常常没有那个 Scripts 目录，或者指向另一个
    venv 的同名脚本（torch 版本都可能不同）。``-m`` 形式只依赖
    ``sys.executable`` 自己的 site-packages —— torch 装了就一定能用。

    为什么单卡不套 torchrun
    ----------------------
    torchrun 在训练进程外面多套一层 agent 进程，改变三件 supervisor 依赖的事：
    信号传递（pause 走 CTRL_BREAK_EVENT / SIGINT 要落到训练进程）、退出码语义
    （agent 把 worker 的失败翻译成自己的 rc）、stdout 汇聚（见下）。单卡用户
    没有任何理由承担这些风险，所以 ``nproc<=1`` 时本函数返回空列表，命令行与
    改动前**逐字节相同**。

    stdout 的 ``__EVENT__:`` 协议怎么保住（最容易出错的一点）
    -------------------------------------------------------
    supervisor 靠 run.log 里 ``line.startswith("__EVENT__:")`` 做进度上报与
    暂停握手（`core._make_task_log_callback`）。torchrun 的 agent 在开启输出
    转发时会给每行**加前缀**（形如 ``[default0]:``，具体模板随 torch 版本与
    ``TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE`` 变化）。一旦 rank 0 的行被加了
    前缀，`startswith` 全部失配 —— 训练照常跑，但进度条不动、暂停按钮永远灰，
    且日志里看不出任何异常，极难排查。

    规避方式不是去猜前缀长什么样，而是让 **rank 0 完全不经 agent 转发**：
    ``--tee`` / ``--redirects`` 都支持 ``<local_rank>:<std>`` 逐 rank 映射，
    **没列出的 rank 一律 Std.NONE**，此时 worker 直接继承 agent 的 stdout fd
    （= supervisor 传进来的 run.log），中间没有任何 Python 层碰过这些字节。
    所以只给 1..N-1 挂 ``--tee``，0 号一个字都不列。这条不依赖前缀模板、不依赖
    torch 版本，是版本无关的不变式。

    非 0 rank 用 ``--tee`` 而不是 ``--redirects``：多卡跑挂时报错常常只出现在
    某个非 0 rank 上，纯 redirect 会让 task log 里只剩一个退出码（trainer 没加
    ``@record``，torchrun 的失败摘要拿不到 traceback），用户在 UI 上看不到原因。
    tee 的行必然以 ``[`` 开头，永远不会被误判成事件行。

    ``--log_dir`` 指进 task 档案（``tasks/<id>/dist/``），让非 0 rank 的
    stdout/stderr 文件跟 run.log / monitor / samples 同根、用户找得到；不给这个
    参数 torchrun 会落到 ``tempfile.mkdtemp()`` 的随机目录，等于丢了。
    """
    if nproc <= 1:
        return []
    args = [
        "-m", "torch.distributed.run",
        # nnodes 默认就是 1，显式写出来是为了让「本项目只支持单机多卡」这件事
        # 在命令行上可见（多机要额外的 rendezvous 编排，不在本改动范围）。
        "--nnodes", "1",
        "--nproc_per_node", str(nproc),
        "--master_port", str(_free_master_port()),
        # 见 docstring：只列 1..N-1，rank 0 留空 = Std.NONE = 原样透传。
        "--tee", ",".join(f"{i}:3" for i in range(1, nproc)),
    ]
    if task_id is not None:
        args += ["--log_dir", str(task_dir(int(task_id)) / "dist")]
    return args


def _default_cmd_builder(task: dict[str, Any], config_path: Path) -> list[str]:
    """根据 task_type 路由到对应脚本。

    train (默认 / 老 task): runtime/anima_train.py
    reg_ai: runtime/anima_reg_ai.py（先验生成）
    generate: 走 inference_daemon，**不**经这个 cmd_builder，supervisor
        在 _dispatch_exclusive_tasks 里直接派给 daemon。这里 fallback 到 anima_generate.py
        只是为了某天测试可能注入 cmd_builder 时不爆 KeyError —— 实际跑
        不到这条 path（_next_pending_task_in 在 dispatch_train 里只挑
        train/reg_ai）。
    """
    task_type = task.get("task_type") or "train"
    if task_type == "reg_ai":
        script = REPO_ROOT / "runtime" / "anima_reg_ai.py"
    elif task_type == "generate":
        script = REPO_ROOT / "runtime" / "anima_generate.py"  # 兜底，正常路径不来这
    else:
        script = REPO_ROOT / "runtime" / "anima_train.py"
    # 只有 train 走多卡：reg_ai（先验生成）与 generate 是单进程推理，DDP 对它们
    # 无意义，而且 generate 正常路径根本不经这里（走 inference_daemon）。
    launcher = (
        _torchrun_prefix(_read_ddp_num_processes(config_path), task.get("id"))
        if task_type == "train"
        else []
    )
    cmd = [
        sys.executable,
        *launcher,  # 单卡时为空 → 命令行与改动前逐字节相同
        str(script),
        "--config",
        str(config_path),
    ]
    msp = task.get("monitor_state_path")
    if msp:
        cmd.extend(["--monitor-state-file", str(msp)])
    # ADR 0006 PR-3: paused task 复活 → 注入 --resume-state，让 anima_train
    # 的 resume_phase 加载 state；旁边的 .config.json snapshot 由 bootstrap_phase
    # 自动检测并 freeze args（ADR §5.7）。
    paused_state = task.get("paused_state_path")
    if paused_state:
        cmd.extend(["--resume-state", str(paused_state)])
    return cmd


def _resolve_monitor_state_path(task: dict[str, Any]) -> Path:
    """决定 task 的 monitor_state.json 落盘路径。

    一律落 `studio_data/tasks/<id>/monitor/state.json`，跟 version 解耦：
    - 删 version 不再带走 task 历史（loss 曲线 / 采样图）
    - 同一 version 多次跑的 task 各自独立档案，用户可以拉出来对比

    历史路径仅保留**读**兼容（老 task DB monitor_state_path 列保留旧值，
    读端按值取，新 task 不再写这些路径）：
    - `versions/<label>/monitor/task_<id>/state.json` —— PP6.1（v0.5.0+）
    - `versions/<label>/monitor_state.json` —— pre-PP6.1
    - `studio_data/monitors/task_<id>/state.json` —— 无 version_id 兜底
    """
    return task_monitor_state_path(task["id"])


def _default_job_cmd_builder(job: dict[str, Any]) -> list[str]:
    """默认按 kind 选 worker 模块。"""
    kind = job["kind"]
    return [
        sys.executable,
        "-m",
        f"studio.workers.{kind}_worker",
        "--job-id",
        str(job["id"]),
    ]
