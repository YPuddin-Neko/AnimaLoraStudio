"""多卡暂停的文件通道。

背景（真机实测）：torchrun 下进程树是 ``supervisor → elastic agent → N × worker``，
supervisor 只持有 agent 的 pid。发 SIGINT 给 agent，它接住后**转发 SIGTERM** 给
worker 并自己抛 SignalException —— worker 的 ``handle_interrupt`` 只注册了
SIGINT/SIGBREAK，收不到 SIGTERM，于是 pause snapshot 不会被写出来、任务无法续训。
日志里能看到 agent 的 "Sending process ... closing signal SIGTERM"。

文件标记绕开信号语义，且让所有 rank 在同一个循环位置看到请求 —— 信号做不到这种
同步：各 rank 收到的时刻不同，一个已 ``sys.exit`` 而另一个还等在 all_reduce 上会
挂住整组。
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, ".")

from utils import distributed as d  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("LORA_TASK_DIR", raising=False)
    yield


def test_no_marker_path_without_task_dir():
    """裸 CLI 训练（无 LORA_TASK_DIR）→ 路径为 None，轮询短路。

    CLI 用 Ctrl+C，信号那条路在单进程下工作正常，不需要文件通道。返回 None 让
    每步的轮询变成一次 dict 查找，开销可忽略。
    """
    assert d.pause_marker_path() is None
    assert d.pause_requested() is False


def test_marker_path_derived_from_task_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("LORA_TASK_DIR", str(tmp_path))
    p = d.pause_marker_path()
    assert p is not None
    assert p.parent == tmp_path
    assert p.name == d.PAUSE_MARKER_NAME


def test_pause_requested_reflects_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("LORA_TASK_DIR", str(tmp_path))
    assert d.pause_requested() is False
    (tmp_path / d.PAUSE_MARKER_NAME).write_text("1", encoding="utf-8")
    assert d.pause_requested() is True


def test_clear_removes_marker(monkeypatch, tmp_path):
    """rank 0 退出前必须删掉标记。

    不删的话下次 resume 会立刻又看到标记、立刻又暂停 —— 表现为「一点继续就又停了」，
    而且用户会以为是 resume 坏了。
    """
    monkeypatch.setenv("LORA_TASK_DIR", str(tmp_path))
    marker = tmp_path / d.PAUSE_MARKER_NAME
    marker.write_text("1", encoding="utf-8")
    d.clear_pause_marker()
    assert not marker.exists()
    assert d.pause_requested() is False


def test_clear_is_idempotent(monkeypatch, tmp_path):
    """标记不存在时删除不该抛 —— 非 rank 0 可能先删过，或用户手工删了。"""
    monkeypatch.setenv("LORA_TASK_DIR", str(tmp_path))
    d.clear_pause_marker()
    d.clear_pause_marker()


def test_clear_without_task_dir_is_noop():
    d.clear_pause_marker()


def test_pause_requested_survives_unreadable_dir(monkeypatch, tmp_path):
    """目录被删/权限问题时按「没有请求」处理，不能抛。

    这是每个训练步都跑的热路径，抛异常会把一次 IO 抖动升级成训练崩溃。
    """
    missing = tmp_path / "gone"
    monkeypatch.setenv("LORA_TASK_DIR", str(missing))
    assert d.pause_requested() is False


def test_marker_name_is_shared_constant():
    """supervisor 与训练侧必须用同一个常量，不能各写一份字符串。

    两边写死不同的文件名 = 暂停永远不生效，而且没有任何报错（supervisor 以为发出去
    了、训练侧永远看不到）。这条钉住 supervisor 是 import 而非复制。
    """
    import pathlib

    src = pathlib.Path("studio/supervisor/core.py").read_text(encoding="utf-8")
    assert "from utils.distributed import PAUSE_MARKER_NAME" in src
    assert f'"{d.PAUSE_MARKER_NAME}"' not in src, (
        "supervisor 里出现了写死的标记文件名 —— 应该只用 import 来的常量"
    )


# ---------------------------------------------------------------------------
# 训练循环的响应位置
# ---------------------------------------------------------------------------


def test_loop_checks_pause_only_at_group_end():
    """轮询必须在**组末步**才响应，不能在梯度累积中间退出。

    mid-accumulation 退出会留下悬挂的 partial backward 梯度（与 ADR 0006
    Addendum 1 放弃 mid-epoch save 的理由同源）。等到组末再响应，最多多跑
    grad_accum-1 个 micro-batch。

    静态检查：条件里必须同时含 _is_group_end_pre 与 pause_requested。
    """
    import pathlib
    import re

    src = pathlib.Path("runtime/training/loop.py").read_text(encoding="utf-8")
    m = re.search(r"if [^\n]*pause_requested\(\)[^\n]*:", src)
    assert m, "loop.py 里没有 pause_requested() 的轮询"
    cond = m.group(0)
    assert "_is_group_end_pre" in cond, (
        "暂停轮询没和组末判定绑定 —— mid-accumulation 退出会留悬挂梯度"
    )
    assert "not ctx.interrupted" in cond, (
        "没挡已 interrupted 状态 —— handle_interrupt 二次触发会走强退分支 exit(1)，"
        "把正常暂停变成失败退出"
    )


# ---------------------------------------------------------------------------
# 启动期检查点
# ---------------------------------------------------------------------------


def test_exit_if_pause_requested_noop_without_marker(monkeypatch, tmp_path):
    """没有标记时什么都不做 —— 每个 phase 之间都会调它，不能有副作用。"""
    monkeypatch.setenv("LORA_TASK_DIR", str(tmp_path))
    d.exit_if_pause_requested()  # 不抛、不退出即通过


def test_exit_if_pause_requested_exits_zero(monkeypatch, tmp_path):
    """见到标记 → SystemExit(0)。

    退出码必须是 0 而不是非零：非零会让 supervisor 的失败摘要把它当训练崩溃处理，
    而这是用户主动要求的停止。cancel_pending 已由 supervisor 置上，收尾走 canceled。
    """
    monkeypatch.setenv("LORA_TASK_DIR", str(tmp_path))
    (tmp_path / d.PAUSE_MARKER_NAME).write_text("1", encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        d.exit_if_pause_requested()
    assert ei.value.code == 0


def test_exit_if_pause_requested_clears_marker_and_emits(monkeypatch, tmp_path):
    """rank 0 退出前删标记并把原因写进 task log。

    删标记是为了下次 resume 不会立刻又停（与训练循环那条同源）。emit 的消息要说清
    「无 epoch 末备份 → 标记为已取消」，否则用户看到任务变 canceled 会以为出错了。
    """
    monkeypatch.setenv("LORA_TASK_DIR", str(tmp_path))
    marker = tmp_path / d.PAUSE_MARKER_NAME
    marker.write_text("1", encoding="utf-8")
    seen: list[str] = []
    with pytest.raises(SystemExit):
        d.exit_if_pause_requested(seen.append)
    assert not marker.exists(), "rank 0 退出前必须删标记"
    assert seen and "取消" in seen[0], f"emit 的消息没说明后果: {seen}"


def test_exit_if_pause_requested_non_main_does_not_clear(monkeypatch, tmp_path):
    """非 rank 0 不删标记 —— 竞态理由同训练循环那条。

    若非 rank 0 先删，rank 0 还没走到检查点就看不见标记、继续往下跑，而其他 rank
    已经退出 → rank 0 卡在下一个集合操作上。
    """
    monkeypatch.setenv("LORA_TASK_DIR", str(tmp_path))
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    marker = tmp_path / d.PAUSE_MARKER_NAME
    marker.write_text("1", encoding="utf-8")
    with pytest.raises(SystemExit):
        d.exit_if_pause_requested()
    assert marker.exists(), "非 rank 0 不该删标记"


def test_startup_checkpoints_between_every_phase():
    """每个长 phase 之后都要有检查点。

    真机上停止请求落在「加载文本编码器」那段（bootstrap 到进训练循环之间），当时
    一个检查点都没有 —— 用户只能取消，而取消在 torchrun 下会抛 SignalException 加
    30 行 traceback，看着像崩溃。

    断言 phase 调用与检查点在 main() 里交替出现，而不只是「存在至少一个」。
    """
    import pathlib
    import re

    src = pathlib.Path("runtime/anima_train.py").read_text(encoding="utf-8")
    body = src[src.find("def main("):]
    # 抓 phase 调用与检查点，按出现顺序
    seq = [
        ("phase", m.group(1)) if m.group(1) else ("check", "")
        for m in re.finditer(
            r"phases\.(\w+)\.(?:run|finish)\(ctx\)|exit_if_pause_requested\(", body,
        )
    ]
    kinds = [k for k, _ in seq]
    # 前四个 phase（bootstrap/models/dataset/text_cache）后面各要跟一个检查点
    covered = {
        name for (k, name), (nk, _) in zip(seq, seq[1:]) if k == "phase" and nk == "check"
    }
    for required in ("bootstrap", "models", "dataset", "text_cache"):
        assert required in covered, (
            f"{required} phase 之后没有停止检查点 —— 该 phase 在真机上是分钟级，"
            f"这段时间用户无法干净地停止训练"
        )
    assert kinds.count("check") >= 4


def test_loop_clears_marker_on_rank0_only():
    """只有 rank 0 删标记。

    非 rank 0 也删的话会出现竞态：rank 1 先删、rank 0 还没轮询到，于是 rank 0
    这一步看不见标记、继续训练，而 rank 1 已经退出 —— rank 0 卡死在下一次
    all_reduce 上。
    """
    import pathlib

    src = pathlib.Path("runtime/training/loop.py").read_text(encoding="utf-8")
    idx = src.find("clear_pause_marker")
    assert idx > 0, "loop.py 没调 clear_pause_marker"
    window = src[max(0, idx - 300):idx]
    assert "is_main()" in window, "clear_pause_marker 没被 is_main() 门控"
