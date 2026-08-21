"""TrainingContext：所有 phase 共享的状态包（ADR 0003 PR-B）。

把原 runtime/anima_train.py 793 行 main() 里的所有 local 变量收到一个 dataclass，
让 phase 函数能 take ctx → mutate → return None 这种风格走流水线。

设计原则：
- 字段类型清晰；late-populated 的用 `Optional[X] = None` 显示
- 进度展示、信号处理等带闭包的逻辑收到本类的方法上（emit / handle_interrupt /
  get_next_sample_prompt），避免 main() 里的 nonlocal 闭包
- 不持有 args 之外的"输入"——任何 yaml / cli 行为都先 merge 进 args，再开始 phase

每个 phase 函数签名：`def run(ctx: TrainingContext) -> None`（in-place mutate）。
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch

# 有意 import **模块对象**而不是 `from utils.distributed import is_main`：后者把函数
# 绑成本模块的全局名，测试 monkeypatch `utils.distributed.is_main` 就打不中了。
# 与 training/dataset.py 的 DDP 分片段同一约定。
from utils import distributed as dist_env

if TYPE_CHECKING:
    from training.losses.protocol import LossProtocol

# ctx.emit 在非 tty 下的出口：采样 / 保存 / resume / 暂停等 user-facing 提示
_emit_logger = logging.getLogger("training.emit")


@dataclass
class TrainingContext:
    # ─── bootstrap_phase 填充 ───
    args: Any  # argparse.Namespace
    family: Any = None  # ModelFamily（多模型 PR-2b；resolve_family(args) 产物）
    config_path: Optional[Path] = None
    config_dir: Optional[Path] = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    # VAE 工作精度，独立于训练 dtype。fp16 训练时 VAE 仍走 fp32：V100/Turing 等需要
    # fp16 的卡没有 bf16，而 fp16 VAE encode/decode 易溢出成 NaN（黑图 / latent 全 NaN
    # → 训练步全跳）。bootstrap 据 dtype 推导；对齐 ComfyUI vae_dtype() 与出图侧 vae_precision。
    vae_dtype: torch.dtype = torch.float32
    output_dir: Optional[Path] = None
    sample_dir: Optional[Path] = None
    wandb_monitor: Any = None         # observability.WandBMonitor
    monitor_server: Optional[bool] = None  # 旧名兼容：True=monitor_state.json 写入活跃
    # supervisor 启动训练时通过 env LORA_TASK_ID 注入 queue task id；CLI 直接跑
    # 时 env 不存在 → None → state_dir() fallback 到 task_unknown 子目录。
    # 注意：跟 progress bar 的 task_id 字段（line ~80）是两回事，故意起不同名字。
    lora_task_id: Optional[int] = None
    # task 档案根（studio_data/tasks/<id>/）。bootstrap 从 --monitor-state-file
    # 上跳两层推出；纯 CLI 没传 → None。samples/ state/ 和 prompt 文本缓存
    # （.text-cache/）都挂在这个根下。
    task_archive_dir: Optional[Path] = None
    # ADR 0006 Addendum 2：auto_epoch_state.pt 落 task 档案（studio_data/tasks/
    # <id>/state/）。bootstrap 从 --monitor-state-file 推出档案根后填充（跟
    # sample_dir 同一约定）；纯 CLI 没传 → None → auto_state_dir() fallback
    # 到 state_dir()。
    task_archive_state_dir: Optional[Path] = None

    # ─── models_phase 填充 ───
    repo_root: Optional[Path] = None
    model: Any = None
    vae: Any = None
    # 文本编码器持有物（family 私有结构，Anima=(qwen_model, qwen_tok, t5_tok)；
    # 对循环 opaque —— 多模型 PR-2b D15，替代原 qwen_model/qwen_tok/t5_tok 三字段）
    text_stack: Any = None
    injector: Any = None
    # DDP 包装器（多卡时由 models_phase._wrap_ddp 填充；单进程恒为 None）。
    #
    # 为什么**不是**把 ctx.model 换成 DDP 对象（这是最容易想错的一步）：
    #   1. `DistributedDataParallel` 不转发属性访问。ctx.model 的下游要读
    #      `.blocks` / `.model_channels` / `.patch_spatial`（SRA、block swap）、
    #      要调 `.train()` / `.eval()`（resume_phase、sample_runner）、还要整个
    #      塞给 family.sample_image() 出图 —— 换成 DDP 对象后全部 AttributeError。
    #   2. checkpoint 的 `module.` 前缀问题从根上不存在：LoRA 产物由
    #      `injector.state_dict()` 出（LyCORIS network 是 DiT 的**兄弟**对象，不是
    #      子模块），ctx.model 保持裸模型意味着没有任何一处 state_dict 会被
    #      DDP 的命名空间污染。
    # 所以：ctx.model 永远是裸模型，DDP 包装器单独放这里，**只有训练前向**走它
    # （loop._forward_via_ddp）—— 因为梯度同步是 DDP.forward 装上去的，不进它
    # 就一次 all_reduce 都不会发生（静默各 rank 独立训练，最坏的失败形态）。
    ddp_model: Any = None  # torch.nn.parallel.DistributedDataParallel (optional)

    # ─── dataset_phase 填充 ───
    bucket_mgr: Any = None
    base_dataset: Any = None
    dataset: Any = None
    reg_dataset: Any = None
    use_cached: bool = False
    dataloader: Any = None

    # ─── optimizer_phase 填充 ───
    weight_decay: float = 0.0
    optimizer: Any = None
    optimizer_type: str = "adamw"
    grad_clip: float = 0.0
    trainable_params: list = field(default_factory=list)
    steps_per_epoch: Optional[int] = None
    total_steps: Optional[int] = None
    # 进度显示专用：每 epoch 按实际包数修正的总步数（navit 打包下 total_steps 是
    # epoch-0 快照会漂移）。只喂 monitor/CLI 进度，scheduler/adapter 仍用 total_steps。
    total_steps_display: Optional[int] = None
    scheduler: Any = None
    timestep_sampler: Any = None    # training.timestep_samplers.TimestepSamplerProtocol
    loss_fn: Optional["LossProtocol"] = None
    sra_aligner: Any = None         # training.families.anima.sra_align.SRAAligner (optional)
    block_swap: Any = None          # training.block_swap.PinnedBlockSwap (optional)

    # ─── resume_phase 填充 ───
    global_step: int = 0
    start_epoch: int = 0
    current_epoch: int = 0
    loss_history: list = field(default_factory=list)
    speed_ema: Optional[float] = None
    progress: Any = None
    task_id: Any = None
    use_rich: bool = False
    use_plain: bool = False
    live: Any = None
    sample_prompts: list = field(default_factory=list)
    sample_prompt_idx: int = 0
    interrupted: bool = False
    # 多卡下信号只置这个标志，不立即退出 —— 见 request_pause_from_signal。
    # 训练循环的轮询点把它和文件标记一起看（loop.py 的两处 pause 检查）。
    pause_signal_seen: bool = False

    # ─── 采样周期的 rank 不变副本（多卡专用）───
    #
    # bootstrap 会在非 rank 0 上把 ``args.sample_steps`` / ``args.sample_every``
    # 置 0，用来关掉「对外输出」（多 rank 往同一个 step_N.png 并发写必然写坏，
    # 而采样是纯推理，多跑 N-1 份纯属浪费）。那个抑制本身是对的。
    #
    # 但采样调用点外面还套着一对 :func:`utils.distributed.barrier`，而 barrier 是
    # **集合操作 —— 必须所有 rank 都执行**。如果 barrier 的门控条件读的是被 rank
    # 改过的 args 字段，非 rank 0 会连 barrier 一起跳过，于是：
    #
    # 1. rank 0 多执行了 N 次 barrier，NCCL 按**调用顺序**配对集合操作，从此每个
    #    rank 的集合操作错位。真机上表现为 rank 1 的 ``_agree_on_finite_loss``
    #    与 rank 0 的 barrier 配上对，读回垃圾值 → 误报「其他 rank 的 loss 非有限值」。
    # 2. rank 0 采样时其余 rank 不再被挡，径直冲进下一步前向；被误报的 NaN skip 又
    #    让上一个 micro-batch 的计算图活着不释放 → 两份 checkpoint 图叠在一起 → OOM。
    #
    # 所以周期值要在**任何 rank 相关改写之前**存一份到 ctx，barrier 的门控只读这份。
    # 「采不采样」由 ``is_main()`` 决定（barrier 里侧），「进不进这个 if」由这份
    # rank 不变副本决定 —— 两件事分开，条件才真的各 rank 一致。
    sample_steps_all_ranks: int = 0
    sample_every_all_ranks: int = 0

    # ─── loop.py epoch backup（ADR 0006 Addendum 1 方案 Δ）───
    # 每 epoch 末尾覆盖式写 auto_epoch_state.pt 后填充这两个字段。
    # handle_interrupt 读它们 emit pause_state event；None 表示首 epoch 还没结束
    # → supervisor 标 canceled 而非 paused（无可恢复进度）。
    last_auto_epoch_state_path: Optional[Path] = None
    last_auto_epoch_config_path: Optional[Path] = None
    scaler: Any = None  # torch.cuda.amp.GradScaler，仅 fp16 时非 None

    # ─── 共用方法 ───

    def state_dir(self) -> Path:
        """用户周期 save（save_state_every*）写 state 的目录，per-task 隔离。

        ADR 0006 §5.3：同一 version 下多 task 跑 state 文件互相覆盖是 latent
        bug，加 task_id 子目录隔离。env LORA_TASK_ID 没设（CLI 直接跑）时
        fallback 到 task_unknown/。
        """
        assert self.output_dir is not None, "state_dir() called before bootstrap_phase"
        tid = self.lora_task_id if self.lora_task_id is not None else "unknown"
        d = self.output_dir / "state" / f"task_{tid}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def auto_state_dir(self) -> Path:
        """auto_epoch_state.pt（系统级恢复点）的落盘目录（ADR 0006 Addendum 2）。

        跟用户周期 save 分家：auto backup 是 task 档案的一部分（同 run.log /
        monitor/ / samples/），落 `studio_data/tasks/<id>/state/` —— 生命周期
        跟 task 行绑定（删 task 一并清），且服务端可从 task id 直接推算路径。
        用户周期 save 是用户产物，留在 state_dir()（version output 树，
        ResumeFieldPicker 按 version 扫得到）。

        纯 CLI（没传 --monitor-state-file）fallback 到 state_dir()，行为不变。
        """
        if self.task_archive_state_dir is not None:
            self.task_archive_state_dir.mkdir(parents=True, exist_ok=True)
            return self.task_archive_state_dir
        return self.state_dir()

    def emit(self, msg: str, level: str = "info") -> None:
        """打印一条 user-facing 消息，按当前进度显示模式分流。

        tty 交互（rich live / progress / plain）三路不变；非 tty（studio spawn 的
        pipe）走 logger，与 run.log 其它行同契约（设计 D3：进度/提示也是日志）。

        ``level``（日志改写 F2）：非 tty 出口按此级别派发到 training.emit logger
        的对应方法。告警性质的 emit（暂停无恢复点、监控历史恢复失败等）不该以
        INFO 混在叙事行里；tty 三路本就没有级别概念，原样打印。

        **多卡下只有 rank 0 真的打印**。门控收在这一处而不是逐个调用点：emit 的
        调用方遍布 loop / resume / finalize / sample_runner，逐处加判断必然漏
        （resume 的「从断点恢复训练」/「采样中 (step 0, 基线)」就漏过），收在入口
        才真正做到日志不 ×N。单进程 ``is_main()`` 恒真，等于这个分支不存在。
        """
        if not dist_env.is_main():
            return
        if self.use_plain:
            print()  # 冲掉 loop 的 `\r` 进度行（tty 交互）
        if self.live:
            self.live.console.print(msg)
        elif self.use_rich:
            self.progress.console.print(msg)
        elif self.use_plain:
            print(msg)
        else:
            getattr(_emit_logger, level, _emit_logger.info)(msg)

    def get_next_sample_prompt(self) -> str:
        """取下一个采样提示词（轮换；sample_prompts 为空则返回默认）。"""
        if not self.sample_prompts:
            return "1girl, masterpiece"
        prompt = self.sample_prompts[self.sample_prompt_idx % len(self.sample_prompts)]
        self.sample_prompt_idx += 1
        return prompt

    def handle_interrupt(self, sig, frame) -> None:
        """Pause / Ctrl+C 信号处理（ADR 0006 Addendum 1 方案 Δ）：Pause = Cancel + 立即释放 GPU。

        信号来源：
          - CLI Ctrl+C：POSIX SIGINT / Windows SIGBREAK（由 resume phase 注册）
          - Supervisor pause：Windows CTRL_BREAK_EVENT / POSIX SIGINT

        新流程（不再 mid-epoch save）：
          1. wandb finish（让 supervisor 读到事件时一切 IO 已完成）
          2. emit __EVENT__:pause_state，state_path 指向**最近一次 epoch 末** auto_epoch_state.pt
             （由 loop.py 每 epoch 末尾覆盖式写盘，ctx.last_auto_epoch_state_path 字段维护）
          3. 首 epoch 内（last_auto_epoch_state_path is None）→ emit state_path=None，
             supervisor 据此走 cancel 分支（ADR 0006 Addendum 1 决策第 3 条）
          4. sys.exit(0)

        放弃 mid-epoch save 的理由（详见 ADR Addendum 1 三方 audit）：
          - grad_accum 周期未守 → partial backward grad 悬挂
          - dataloader 进度不存 → resume 5% double-train（Prodigy d 估计偏）
          - current_epoch 语义二义性（mid-epoch 路径保 epoch / epoch-end 路径保 epoch+1）
          - InfoNoise / cosine restart T_cur 漂移
          - 真正符合"暂停 = 立即释放 GPU"产品语义

        重复触发（已 interrupted 状态再来一次）= 强退。
        """
        # 延迟 import 避免循环依赖
        from studio.infrastructure.log_messages import msg
        from training.snapshot import emit_event

        if self.interrupted:
            self.emit(msg("train.force_exit"))
            sys.exit(1)
        self.interrupted = True
        self.emit(msg("train.pause_signal"))
        try:
            self.wandb_monitor.finish()
        except Exception:
            pass
        # emit 在 wandb finish 后 — 让 supervisor 读到事件时一切 IO 已完成。
        #
        # 多卡：`__EVENT__:` 只能由 rank 0 写。supervisor 只读一个 stdout，N 个 rank
        # 同写 pause_state 会让它按 N 次暂停处理（第二次起 state_path 指向同一份
        # 文件、step 却可能不同）—— 表现为任务状态在 paused/canceled 之间反复跳。
        # 非 rank 0 仍然照常 sys.exit(0)：torchrun 见到任一子进程退出会收掉整组，
        # 所以「只有 rank 0 上报、所有 rank 都退出」是正确组合。
        if dist_env.is_main():
            emit_event("pause_state", {
                "state_path": str(self.last_auto_epoch_state_path) if self.last_auto_epoch_state_path else None,
                "config_path": str(self.last_auto_epoch_config_path) if self.last_auto_epoch_config_path else None,
                "step": self.global_step,
            })
        if self.last_auto_epoch_state_path:
            self.emit(msg("train.paused_with_state", path=self.last_auto_epoch_state_path))
        else:
            self.emit(
                "Pause without a resume state: the first epoch never finished, so no "
                "epoch backup exists — the task is marked canceled and this run's "
                "progress is discarded",
                level="warning",
            )
        sys.exit(0)

    def request_pause_from_signal(self, signum, frame) -> None:  # noqa: ARG002
        """多卡下的信号处理器：**只置标志**，不退出。

        为什么不能像单卡那样直接 ``handle_interrupt``：那会让收到信号的 rank 单独
        走掉，其余 rank 继续阻塞在下一个集合操作上等它，直到被 torchrun SIGKILL
        （真机 epoch 9 暂停就是这么挂的，详见 phases/resume.py 里信号注册处的注释）。

        置了标志之后由训练循环的轮询点响应 —— 那些位置所有 rank 都到得了，
        于是「一起看到、一起退出」。代价是响应延迟：最坏情况要等当前采样跑完
        （3-7 分钟）。UI 那个 30 秒 modal 会先弹「保存耗时超过预期」，
        文案已按此调整。

        重复按（标志已置）= 用户等不下去了，按单卡语义立即强退。这条路径下
        各 rank 状态本就要丢，不再顾虑集合操作。
        """
        if self.pause_signal_seen:
            self.emit("再次收到暂停信号 → 强制退出（放弃本次 epoch 的干净收尾）...")
            sys.exit(1)
        self.pause_signal_seen = True
        self.emit(
            "收到暂停信号，将在下一个安全点退出（多卡需所有 rank 同步；"
            "若正在采样要等它跑完）..."
        )
