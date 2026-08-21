"""resume_phase：progress 初始化 + state recovery + 信号注册 + step 0 baseline 采样。

抽自 main() L439-594（ADR 0003 PR-B）。
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

from studio.infrastructure.log_messages import msg
from training.bootstrap import init_progress
from training.context import TrainingContext
from training.observability import render_curve_panel
from training.sample_runner import run_sample
from utils import distributed as dist_env
from training.snapshot import emit_event
from training.state import load_training_state


def run(ctx: TrainingContext) -> None:
    """
    - init_progress + 可选 Rich Live（含 loss curve panel）
    - 如有 --resume-state：load_training_state + restore monitor 历史 loss
    - 注册 SIGINT → ctx.handle_interrupt
    - 准备 sample_prompts 列表（多角色轮换）
    - global_step==0 时跑 baseline 采样（最多 3 prompt）
    """
    args = ctx.args

    # 初始化进度显示
    ctx.progress, ctx.task_id, progress_kind = init_progress(not args.no_progress, ctx.total_steps)
    ctx.use_rich = progress_kind == "rich"
    ctx.use_plain = ctx.progress == "plain"
    ctx.live = None
    ctx.loss_history = []
    ctx.speed_ema = None

    if ctx.use_rich:
        try:
            from rich.console import Group
            from rich.live import Live
            curve_panel = None
            if args.loss_curve_steps > 0 and not args.no_live_curve:
                curve_panel = render_curve_panel([], width=min(60, args.loss_curve_steps), height=10)
            group = Group(ctx.progress, curve_panel) if curve_panel is not None else Group(ctx.progress)
            ctx.live = Live(group, refresh_per_second=10)
            ctx.live.start()
        except Exception:
            ctx.live = None
            ctx.progress.start()

    # 训练循环初始状态
    ctx.global_step = 0
    ctx.start_epoch = 0

    # 从训练状态恢复（断点续训）
    if getattr(args, "resume_state", "") and Path(args.resume_state).exists():
        ctx.start_epoch, ctx.global_step, ctx.loss_history, saved_monitor_state = load_training_state(
            args.resume_state, ctx.injector, ctx.optimizer, ctx.scheduler,
            timestep_sampler=ctx.timestep_sampler,
            sra_aligner=ctx.sra_aligner,
            scaler=ctx.scaler,
            expected_family=ctx.family.spec.family_id,
        )
        # resume 的叙事行由 load_training_state 打（state.py），此处不重复。

        # 恢复监控面板的历史数据（loss 曲线等）
        if ctx.monitor_server and saved_monitor_state:
            try:
                from train_monitor import restore_monitor_state
                restore_monitor_state(
                    losses=saved_monitor_state.get("losses"),
                    lr_history=saved_monitor_state.get("lr_history"),
                    optimizer_metrics_history=saved_monitor_state.get("optimizer_metrics_history"),
                    epoch=ctx.start_epoch,
                    step=ctx.global_step,
                    total_steps=ctx.total_steps,
                )
                ctx.emit(msg(
                    "train.monitor_history_restored",
                    n=len(saved_monitor_state.get("losses", [])),
                ))
            except Exception as e:
                ctx.emit(
                    f"Dashboard history could not be restored: {e} — the loss chart "
                    f"starts from this step, training itself is not affected",
                    level="warning",
                )

        # ADR §`_on_line` 识别此事件后清理上次 pause 文件对（PR-3 cmd_builder 接入）。
        emit_event("resume_state_loaded", {"path": str(args.resume_state)})

    # 信号处理：handle_interrupt 由 TrainingContext 自带，跨平台双绑
    # （ADR §`runtime/training/phases/resume.py`）：
    #   POSIX：SIGINT（CLI Ctrl+C / supervisor `os.kill(pid, SIGINT)`）
    #   Windows：SIGINT 留给 CLI Ctrl+C，SIGBREAK 接 supervisor 发的
    #     CTRL_BREAK_EVENT（CREATE_NEW_PROCESS_GROUP 子进程组收不到 CTRL_C_EVENT）
    #
    # 多卡：信号**不能**直接退出，只置标志，由训练循环里已有的轮询点在
    # 「所有 rank 都到得了的同步位置」响应。
    #
    # 直接退出的真机后果（epoch 9 暂停）：信号在采样期间到达，rank 0 正在
    # is_main() 里出图，handle_interrupt 跑完就 sys.exit(0)；rank 1 早已越过采样前
    # 的 barrier、此刻阻塞在采样后那个 barrier 上等永不到来的 rank 0。NCCL 的
    # barrier 阻塞在 C++ 里，Python 信号处理器要等当前 C 调用返回才有机会跑，
    # 所以 rank 1 连自己的信号都处理不了 —— 30 秒宽限期后被 torchrun SIGKILL
    # （日志：22:13:28 发信号 → 22:13:58 "forcefully exiting via 9"），
    # 死在集合操作中间，RCCL 通信器没正常销毁。
    #
    # supervisor 的文件标记通道（_write_pause_marker）本来就是为多卡设计的，
    # 但它的注释假设「worker 每步轮询」—— 采样一次 3-7 分钟，那不是一个训练步，
    # 期间没有轮询点，所以光有文件通道不够，还必须把信号这条路也改成异步。
    #
    # 单卡保持原样：没有集合操作，立即退出是安全的，且 CLI Ctrl+C 依赖它的即时性。
    if dist_env.world_size() > 1:
        signal.signal(signal.SIGINT, ctx.request_pause_from_signal)
        if os.name == "nt":
            signal.signal(  # type: ignore[attr-defined]
                signal.SIGBREAK, ctx.request_pause_from_signal,  # type: ignore[attr-defined]
            )
    else:
        signal.signal(signal.SIGINT, ctx.handle_interrupt)
        if os.name == "nt":
            # SIGBREAK 在 POSIX 上不存在；只 Windows 注册
            signal.signal(signal.SIGBREAK, ctx.handle_interrupt)  # type: ignore[attr-defined]

    ctx.current_epoch = ctx.start_epoch
    ctx.model.train()
    # Schedule-Free 系优化器（PPSF / soap_sf 等）须从 train_mode 起步：参数张量
    # 持有梯度评估点 y 而非 averaged x。duck-type 而非硬编码 optimizer_type，新增
    # schedule-free 变体零改动；AdamW / Prodigy 无 .train 方法走 hasattr 静默跳过。
    if hasattr(ctx.optimizer, "train") and callable(getattr(ctx.optimizer, "train")):
        ctx.optimizer.train()
    # step_start_time 由 train_loop 内自己重置；这里不需要

    # 设置采样提示词列表（支持多角色轮换）
    ctx.sample_prompts = getattr(args, "sample_prompts", []) or []
    if not ctx.sample_prompts and args.sample_prompt:
        ctx.sample_prompts = [args.sample_prompt]
    ctx.sample_prompt_idx = 0

    # Step 0 初始采样（基线效果，测试所有提示词）
    # 只在新训练时执行（global_step == 0），resume 时跳过
    #
    # 多卡：只 rank 0 出图 + 前后 barrier，与 loop.py 的 step / epoch 采样同口径。
    # 这里**曾经漏了**，后果是真机上 rank 0 还在基线采样、rank 1 已经冲进训练前向
    # 的第一个 attention —— 两者的显存峰值叠在同一时间窗口，rank 1 OOM。
    # （各 rank 还会并发写同一批 step_0_baseline_*.png，必然写坏。）
    #
    # barrier 在 if 里侧、is_main() 外侧：集合操作必须所有 rank 都执行。
    #
    # 门控读 ctx.sample_*_all_ranks 而**不是** args.sample_* —— 后者在非 rank 0 上
    # 被 bootstrap 置 0 了（关掉重复出图），拿它当门控会让 rank 1 连 barrier 一起
    # 跳过。真机上的后果：rank 1 不等 rank 0 采样完就冲进训练前向，而且 NCCL 集合
    # 操作从此错位（rank 1 的 loss all_reduce 与 rank 0 的 barrier 配对，读回垃圾值
    # → 误报「其他 rank 的 loss 非有限值」）。详见 TrainingContext 那两个字段的注释。
    sampling_enabled = (
        ctx.sample_steps_all_ranks > 0 or ctx.sample_every_all_ranks > 0
    )
    if ctx.global_step == 0 and sampling_enabled:
        # barrier 在 if 外侧、is_main() 在里侧 —— 集合操作必须所有 rank 都执行。
        # 采样前也挡一次：让 rank 0 在其余 rank 的激活都已释放之后才开始出图，
        # 否则两者的显存峰值叠在同一时间窗口（真机上直接 OOM）。
        dist_env.barrier()
        if dist_env.is_main():
            ctx.emit(msg("train.baseline_sampling"))
            for i, prompt in enumerate(ctx.sample_prompts[:3]):  # 最多测试 3 个
                sample_path = ctx.sample_dir / f"step_0_baseline_{i}.png"
                run_sample(
                    ctx,
                    prompt=prompt,
                    sample_path=sample_path,
                    wandb_key="samples/baseline",
                    wandb_caption=f"step 0 baseline {i}: {prompt}",
                    wandb_step=0,
                    seed_offset=i,
                )
        dist_env.barrier()
    elif ctx.global_step > 0 and sampling_enabled:
        ctx.emit(msg("train.baseline_sampling_skipped", step=ctx.global_step))

    # ADR §8.1 is_pausable 信号：resume phase 全部跑完 → 训练进入主循环 →
    # 允许用户暂停。supervisor `_on_line` 收到此事件后 slot.train_loop_started = True
    # → 通过 SSE 派发 is_pausable=True 解锁 UI 暂停按钮。
    emit_event("train_loop_started", {"global_step": ctx.global_step})
