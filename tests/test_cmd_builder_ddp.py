"""torchrun 多进程启动分支（cmd_builder）—— 单卡逐字节不变 + 多卡参数正确。

不起子进程、不需要装 torch：`_default_cmd_builder` 只拼字符串列表，多卡分支的
唯一输入是 task 的 config.yaml 里的 `ddp_num_processes`。

本文件锁的最要紧一条是 **`--tee` 里没有 rank 0**（理由见
`_torchrun_prefix` docstring）：torchrun 一旦接管 rank 0 的 stdout 就会给每行加
前缀，supervisor 的 `line.startswith("__EVENT__:")` 随之全部失配 —— 训练照常
跑，但进度不动、暂停按钮永远灰，且日志里看不出任何异常。这条不变式在运行时
没有任何断言兜着，只有这里。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from studio.infrastructure.paths import REPO_ROOT, task_dir
from studio.supervisor.cmd_builder import _default_cmd_builder

TRAIN_SCRIPT = str(REPO_ROOT / "runtime" / "anima_train.py")


def _cfg(tmp_path: Path, body: str = "") -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def _train(cfg: Path, **extra: object) -> list[str]:
    return _default_cmd_builder({"task_type": "train", "id": 7, **extra}, cfg)


# --------------------------------------------------------------- 单卡不变式

@pytest.mark.parametrize(
    "body",
    ["", "ddp_num_processes: 1\n", "batch_size: 2\n", "ddp_num_processes: 0\n"],
)
def test_single_gpu_cmd_is_byte_identical(tmp_path: Path, body: str) -> None:
    """缺键 / 显式 1 / 非法 0 → 命令行与套 torchrun 之前完全相同。

    单卡是绝对多数用户的路径，torchrun 那层 agent 会改信号传递与退出码语义，
    没理由让他们承担。
    """
    cfg = _cfg(tmp_path, body)
    assert _train(cfg) == [sys.executable, TRAIN_SCRIPT, "--config", str(cfg)]


def test_broken_yaml_falls_back_to_single_gpu(tmp_path: Path) -> None:
    """config.yaml 损坏时降级单卡而**不抛** —— `_tick` 的异常被 `_loop` 吞掉
    后按 poll 间隔无限重试，task 会永远停在 pending 且日志被刷爆。"""
    cfg = _cfg(tmp_path, "ddp_num_processes: [not, an, int\n")
    assert _train(cfg) == [sys.executable, TRAIN_SCRIPT, "--config", str(cfg)]


# ----------------------------------------------------------------- 多卡分支

def test_multi_gpu_wraps_torchrun_with_nproc(tmp_path: Path) -> None:
    """>1 → 走 `python -m torch.distributed.run`，nproc_per_node 等于配置值。

    用 `-m` 而不是裸 `torchrun`：后者是 venv 里的 console script，子进程 PATH
    未必指向同一个 venv；`-m` 只依赖 sys.executable 自己的 site-packages。
    """
    cmd = _train(_cfg(tmp_path, "ddp_num_processes: 4\n"))
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "torch.distributed.run"]
    assert cmd[cmd.index("--nproc_per_node") + 1] == "4"
    assert cmd[cmd.index("--nnodes") + 1] == "1"


def test_multi_gpu_keeps_script_and_config_after_launcher(tmp_path: Path) -> None:
    """torchrun 参数全部在训练脚本**之前**，脚本自己的 --config 紧跟其后。

    顺序错了 torchrun 会把 --config 当成自己的参数解析失败，或者把训练脚本
    当成 torchrun 的位置参数。
    """
    cfg = _cfg(tmp_path, "ddp_num_processes: 2\n")
    cmd = _train(cfg)
    si = cmd.index(TRAIN_SCRIPT)
    assert cmd[si + 1: si + 3] == ["--config", str(cfg)]
    assert "--nproc_per_node" in cmd[:si]  # launcher 参数不越过脚本


def test_multi_gpu_tee_excludes_rank0(tmp_path: Path) -> None:
    """**协议关键**：--tee 只列 1..N-1，rank 0 缺席 = Std.NONE = 原样透传。

    rank 0 的 stdout 一旦经 torchrun 转发就会被加 `[default0]:` 之类前缀，
    supervisor 的 `__EVENT__:` 匹配随之全灭。
    """
    cmd = _train(_cfg(tmp_path, "ddp_num_processes: 4\n"))
    tee = cmd[cmd.index("--tee") + 1]
    assert tee == "1:3,2:3,3:3"
    assert not tee.startswith("0:")
    assert "0:" not in tee
    # 也不能出现全局形式（`--tee 3` 会把 rank 0 一起接管）
    assert ":" in tee


def test_multi_gpu_never_redirects_rank0_globally(tmp_path: Path) -> None:
    """不出现无 rank 映射的 --redirects/--tee 全局值（那会覆盖 rank 0）。"""
    cmd = _train(_cfg(tmp_path, "ddp_num_processes: 2\n"))
    assert "--redirects" not in cmd  # 用 tee 保住非 0 rank 报错在 task log 可见
    for flag in ("--tee",):
        val = cmd[cmd.index(flag) + 1]
        assert all(":" in part for part in val.split(","))


def test_master_port_is_dynamic_and_distinct(tmp_path: Path) -> None:
    """每次构建拿一个新的空闲端口，不用默认 29500。

    同机可能有第二个 Studio 实例 / 手跑的 torchrun / 上次崩溃残留的 TIME_WAIT
    socket，固定端口会让后来者直接起不来。
    """
    cfg = _cfg(tmp_path, "ddp_num_processes: 2\n")
    ports = {_train(cfg)[_train(cfg).index("--master_port") + 1] for _ in range(3)}
    for p in ports:
        assert 1024 < int(p) < 65536
        assert p != "29500"


def test_log_dir_points_into_task_archive(tmp_path: Path) -> None:
    """非 0 rank 的 stdout 文件落 tasks/<id>/dist/，跟 run.log 同根。

    不传 --log_dir 时 torchrun 落到 tempfile.mkdtemp() 的随机目录，用户找不到。
    """
    cmd = _train(_cfg(tmp_path, "ddp_num_processes: 2\n"))
    assert cmd[cmd.index("--log_dir") + 1] == str(task_dir(7) / "dist")


def test_multi_gpu_coexists_with_resume_state(tmp_path: Path) -> None:
    """torchrun + --resume-state / --monitor-state-file 同时在时互不干扰
    （ADR 0006 的 resume 链路不能被多卡改动破坏）。"""
    cfg = _cfg(tmp_path, "ddp_num_processes: 2\n")
    cmd = _train(cfg, paused_state_path="/tmp/p.pt", monitor_state_path="/tmp/s.json")
    assert cmd[cmd.index("--resume-state") + 1] == "/tmp/p.pt"
    assert cmd[cmd.index("--monitor-state-file") + 1] == "/tmp/s.json"
    assert cmd.index(TRAIN_SCRIPT) < cmd.index("--resume-state")


@pytest.mark.parametrize("task_type", ["reg_ai", "generate"])
def test_non_train_tasks_never_use_torchrun(tmp_path: Path, task_type: str) -> None:
    """reg_ai（先验生成）/ generate 是单进程推理，DDP 对它们无意义。"""
    cfg = _cfg(tmp_path, "ddp_num_processes: 4\n")
    cmd = _default_cmd_builder({"task_type": task_type, "id": 7}, cfg)
    assert "torch.distributed.run" not in cmd
    assert cmd[0] == sys.executable
