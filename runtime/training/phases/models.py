"""models_phase: paths + family-owned weights + LoRA injection.

Cached-varlen families may defer their large DiT until dataset captions have
been cached and the text encoder released. ``finish(ctx)`` closes that deferred
half immediately after ``text_cache`` and before optimizer construction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from studio.infrastructure.log_messages import msg
from training.context import TrainingContext
from training.families import resolve_family
from training.families.anima import ANIMA_SPEC as _ANIMA_SPEC
from training.sysmem import log_vram
from training.model_loading import (
    find_diffusion_pipe_root,
    resolve_path_best_effort,
)
# 模块对象 import（不是 `from utils.distributed import ...`）：测试要 monkeypatch
# `utils.distributed.*`。同 training/dataset.py 的约定。
from utils import distributed as dist_env


logger = logging.getLogger(__name__)

#: 日志里 lora_type 的规范大小写。裸 ``.upper()`` 会打出「LOKR」这种不成词的
#: 拼写；schema 的 Literal 集合是权威来源，新增变体在此补一行。
_LORA_TYPE_LABELS = {
    "lora": "LoRA",
    "lokr": "LoKr",
    "loha": "LoHa",
    "ortho": "Ortho",
    "tlora": "T-LoRA",
}


def _resolve_paths(ctx: TrainingContext) -> None:
    args = ctx.args
    ctx.repo_root = find_diffusion_pipe_root()
    logger.debug("model_code: path=%s", ctx.repo_root)

    phases_dir = Path(__file__).resolve().parent
    training_dir = phases_dir.parent
    runtime_dir = training_dir.parent
    bases = [
        Path.cwd(),
        ctx.config_dir,
        ctx.config_dir.parent if ctx.config_dir else None,
        runtime_dir,
        runtime_dir.parent,
        ctx.repo_root,
        ctx.repo_root.parent,
    ]
    args.transformer_path = resolve_path_best_effort(args.transformer_path, bases)
    args.vae_path = resolve_path_best_effort(args.vae_path, bases)
    args.text_encoder_path = resolve_path_best_effort(args.text_encoder_path, bases)
    args.t5_tokenizer_path = resolve_path_best_effort(args.t5_tokenizer_path, bases)
    args.data_dir = resolve_path_best_effort(args.data_dir, bases)
    reg_data_dir = getattr(args, "reg_data_dir", "") or ""
    if reg_data_dir:
        args.reg_data_dir = resolve_path_best_effort(reg_data_dir, bases)


def _swap_vram_discount(ctx: TrainingContext) -> float:
    """开 block swap 时不会进显存的权重**比例**（预算折扣，见 check_load_budget）。

    比例而非字节：fp8 与 bf16 文件大小差一倍，按字节折扣会在 fp8 场景折扣穿。
    族未实现估算就返回 0（护栏退化成保守，不会误放行）。
    """
    blocks_to_swap = int(getattr(ctx.args, "blocks_to_swap", 0) or 0)
    if blocks_to_swap <= 0:
        return 0.0
    ratio_fn = getattr(ctx.family, "swapped_param_ratio", None)
    if ratio_fn is None:
        return 0.0
    try:
        # checkpoint_path：anima 靠它区分 28/36 层版本；krea2 结构唯一、忽略
        return float(ratio_fn(
            blocks_to_swap,
            checkpoint_path=str(getattr(ctx.args, "transformer_path", "") or ""),
        ))
    except Exception:  # noqa: BLE001
        return 0.0


def _load_dit(ctx: TrainingContext) -> None:
    args = ctx.args
    backend = getattr(args, "attention_backend", "flash_attn")
    if backend == "none":
        logger.info(msg("train.attention_sdpa"))
    logger.info(msg("train.loading_transformer"))
    extra = {}
    blocks_to_swap = int(getattr(args, "blocks_to_swap", 0) or 0)
    if blocks_to_swap > 0:
        # 能力位在 schema 侧已用 cap_gate 门控；裸 CLI / 旧 yaml 仍可能带上，
        # 这里 fail-fast 而非静默忽略（否则用户以为省了显存其实没有）
        if "block_swap" not in ctx.family.spec.capabilities:
            raise RuntimeError(
                f"model_family='{ctx.family.spec.family_id}' 不支持 block swap，"
                f"但 blocks_to_swap={blocks_to_swap}。请置 0。"
            )
        # 换出层由 loader 直接落 CPU pinned，不经过显存（12/16GB 目标的前提）
        extra["blocks_to_swap"] = blocks_to_swap
        logger.debug(
            "block_swap: planned swap_blocks=%d (tail blocks stay in pinned memory, "
            "never loaded to VRAM)", blocks_to_swap,
        )
    ctx.model = ctx.family.load_dit(
        args.transformer_path,
        ctx.device,
        ctx.dtype,
        attention_backend=backend,
        repo_root=ctx.repo_root,
        **extra,
    )
    # 大权重 mmap 缓存页归还系统（13-26GB；真机换页卡死案例，训练同样受益）
    from training.sysmem import trim_working_set

    trim_working_set()
    log_vram("train.vram_stage_transformer_loaded", ctx.device)


def _log_train_start_vram(ctx: TrainingContext) -> None:
    """训练循环开始前的显存基线 —— 判断 blocks_to_swap 实际效果的读数点。"""
    swap = getattr(ctx, "block_swap", None)
    if swap is not None:
        logger.info(msg(
            "train.block_swap_active",
            n=swap.num_swap, total=swap.total,
            pinned=f"{swap.pinned_bytes / 1024**3:.2f}",
        ))
    log_vram("train.vram_stage_train_start", ctx.device)


def _load_vae(ctx: TrainingContext) -> None:
    args = ctx.args
    logger.info(msg("train.loading_vae"))
    ctx.vae = ctx.family.load_vae(
        args.vae_path,
        ctx.device,
        ctx.vae_dtype,
        tiling=getattr(args, "vae_tiling", "auto"),
    )


def _load_text(ctx: TrainingContext) -> None:
    args = ctx.args
    logger.info(msg("train.loading_text_encoder"))
    ctx.text_stack = ctx.family.load_text(
        args.text_encoder_path,
        ctx.device,
        ctx.dtype,
        t5_tokenizer_path=args.t5_tokenizer_path,
        cache_enabled=bool(getattr(args, "text_encoder_cache", True)),
    )


def _setup_block_swap(ctx: TrainingContext) -> None:
    """构造并挂载 block swap（docs/design/block-swap.md 刀 2）。

    **必须在 LoRA 注入之后**：LyCORIS ``apply_to()`` 会读基权重的 shape 建适配器，
    此时换出层的权重是 loader 落下的 CPU pinned 张量（shape/dtype 完好，只是不在
    显存），注入正常；反过来若先 attach 再注入也可行，但没有理由把顺序搞复杂。

    挂载走 hook（``attach()``），不改模型 forward 循环 —— krea2 的循环在
    parity 敏感的 ``modeling/`` 内，anima 的手工展开循环也不必动。前向 +
    反向四个钩子缺一不可（反向必须自己取回权重，重算不触发 forward_hook，
    见 doc §9.10 与 tests/test_block_swap_grad_fidelity.py）。
    """
    blocks_to_swap = int(getattr(ctx.args, "blocks_to_swap", 0) or 0)
    if blocks_to_swap <= 0:
        return
    from training.block_swap import PinnedBlockSwap

    # clamp 到总层数：blocks_to_swap 是跨族/跨版本共享的设置值（krea2 28 层、
    # anima 28/36 层），超界按全量换出处理而非 fail（loader 侧同口径）
    ctx.block_swap = PinnedBlockSwap(
        ctx.model.blocks, min(blocks_to_swap, len(ctx.model.blocks)), ctx.device,
    )
    ctx.block_swap.attach()
    log_vram("train.vram_stage_block_swap_ready", ctx.device)


def _inject_adapter(ctx: TrainingContext) -> None:
    args = ctx.args
    lora_type = str(args.lora_type)
    logger.info(msg(
        "train.injecting_lora",
        lora_type=_LORA_TYPE_LABELS.get(lora_type, lora_type),
    ))
    from training.adapters import build_adapter

    ctx.injector = build_adapter(args, preset=ctx.family.lora_preset())
    ctx.injector.metadata_extra = ctx.family.lora_metadata()
    ctx.injector.inject(ctx.model)

    if getattr(args, "resume_lora", "") and Path(args.resume_lora).exists():
        lora_family = _read_lora_family(args.resume_lora)
        if lora_family != ctx.family.spec.family_id:
            raise RuntimeError(
                f"resume_lora 跨模型族被拒绝：{args.resume_lora} 属于 '{lora_family}'，"
                f"当前 model_family='{ctx.family.spec.family_id}'"
            )
        ctx.injector.load(args.resume_lora)
        logger.info(msg("train.resume_from_lora", path=args.resume_lora))

    _setup_block_swap(ctx)

    if getattr(args, "sra_enabled", False):
        from training.families.anima.sra_align import SRAAligner

        model_channels = ctx.model.model_channels
        block_idx = int(getattr(args, "sra_block", 4))
        num_blocks = len(ctx.model.blocks)
        if block_idx >= num_blocks:
            logger.warning(
                "sra_block=%d is beyond the model block count (%d): clamped to %d",
                block_idx,
                num_blocks,
                num_blocks - 1,
            )
            block_idx = num_blocks - 1
        ctx.sra_aligner = SRAAligner(
            model=ctx.model,
            block_idx=block_idx,
            patch_spatial=ctx.model.patch_spatial,
            patch_temporal=ctx.model.patch_temporal,
            model_channels=model_channels,
            vae_channels=_ANIMA_SPEC.latent.channels,
            device=ctx.device,
            dtype=ctx.dtype,
            normalize=bool(getattr(args, "sra_normalize", True)),
        )


class _DDPAdapterSync(torch.nn.Module):
    """DDP 真正包住的东西：**只有 adapter 的可训练参数**，外加一个转调前向。

    为什么不是 ``DDP(ctx.model)``（两个独立的硬理由，都是踩过才知道的）
    ------------------------------------------------------------------
    1. **DiT 里没有可训练参数。** LyCORIS 的 ``apply_to()`` 是 monkeypatch
       原模块的 ``forward``，适配器模块自己挂在 ``LycorisNetwork`` 上，并把原模块
       用 ``org_module=[m]``（**列表**，刻意绕开子模块注册）持有。于是
       ``dit.parameters()`` 里全是 frozen 基座，一个 ``requires_grad=True`` 都没有 ——
       ``DDP(dit)`` 会直接抛「not needed when a module doesn't have any parameter
       that requires a gradient」，就算不抛也永远同步不到 LoRA 梯度。
    2. **训练前向压根不走 ``dit.__call__``。** Anima 开梯度检查点时手工展开
       ``prepare_embedded_sequence`` / ``t_embedder`` / ``blocks`` / ``final_layer``
       （族私货，见 families/anima/forward.py），navit 走
       ``forward_packed_navit``。而 DDP 的梯度同步是 ``DDP.forward`` 里
       ``prepare_for_backward`` 装上去的：不进 DDP.forward，reducer 的
       ``expect_autograd_hooks_`` 就一直是 false，所有 autograd hook 直接 return ——
       **一次 all_reduce 都不会发生，且不报任何错**。各 rank 悄悄各训各的、最后存
       rank 0 那份，是本次集成最危险的失败形态。

    所以这里反过来做：把「要同步的参数」和「怎么跑前向」解耦。

    - 参数：``nn.ParameterList(trainable)`` 注册的是**同一批 Parameter 对象**（不复制），
      所以 DDP 的桶、autograd hook、optimizer 三方引用的是同一批张量。传什么进来
      就同步什么 —— 同步集合显式可审计，而不是「模块树里恰好有什么」。这也让
      SRA projection MLP 能被有意排除（它的 loss 在 DDP 前向之外算，硬塞进来会被
      find_unused_parameters 提前 mark ready，反而出错）。
    - DiT：用 ``self._dit = [dit]`` 列表持有（跟 LyCORIS 的 org_module 同一手法），
      **不进模块树**。两个好处：不会和 ParameterList 里的参数重复注册；DDP 构造时的
      ``_sync_module_states`` 只广播几十 MB 的 adapter 参数，而不是把 26GB frozen
      基座在卡间广播一遍（那是几十秒到几分钟的启动停顿，而且毫无必要 —— 每个 rank
      都从同一个 checkpoint 文件读，本来就一样）。
    - 前向：``forward(step_fn, *args)`` 把真正的前向函数当入参收进来再转调，于是
      标准 / navit / 展开检查点三条路径**全都**能进 DDP.forward，无需改 families/。
      callable 经 DDP 的 ``_to_kwargs`` 时落在「非 tensor/list/dict → 原样复制」的
      分支上，不会被搬运或篡改。
    """

    def __init__(self, dit, trainable_params):
        super().__init__()
        self.trainables = torch.nn.ParameterList(list(trainable_params))
        self._dit = [dit]

    def forward(self, step_fn, *args, **kwargs):
        return step_fn(self._dit[0], *args, **kwargs)


def _needs_find_unused_parameters(args) -> bool:
    """本次配置下是否真可能有 adapter 参数不参与某一步的前向。

    只有 ``lora_module_dropout > 0`` 会造成这种情况：LyCORIS 的 stochastic depth
    按**每个模块、每一步**独立掷骰子决定要不要整块跳过，被跳过的模块该步没有梯度，
    而各 rank 的随机数不同步 —— 跳的不是同一批。此时若 ``find_unused_parameters=False``，
    DDP 会等一个永远不来的梯度，抛「Expected to have finished reduction in the
    prior iteration」。

    另外两个 dropout **不算**，区别很关键：
    - ``lora_dropout`` 丢的是**输入特征**（对激活做 mask），参数照常参与矩阵乘，
      梯度照常产生（只是数值上被 mask 影响）。
    - ``lora_rank_dropout`` 丢的是 rank 维度的一部分。LyCORIS 实现是对中间激活乘
      mask，``lora_down`` / ``lora_up`` 两个张量整体仍在计算图里 —— 参数粒度上没有
      「未参与」。DDP 看的是参数粒度，所以不受影响。

    T-LoRA 也不算：它按 timestep 改的是 **rank mask buffer** 的内容，参与前向的
    参数张量集合不变（见 ``broadcast_buffers=False`` 那段说明）。

    SRA 同理不算：它的 projection MLP 压根不在 DDP 的同步集合里（``_DDPAdapterSync``
    只收 ``injector.get_params()``），不存在「注册了但没用」。

    误判成本不对称，所以这里的默认取向是「拿不到配置就开着」：多花点时间总比训练
    崩掉好。
    """
    try:
        return float(getattr(args, "lora_module_dropout", 0.0) or 0.0) > 0.0
    except (TypeError, ValueError):
        return True


def _wrap_ddp(ctx: TrainingContext) -> None:
    """多卡时把 adapter 参数包进 DDP；单进程 no-op（``ctx.ddp_model`` 保持 None）。

    时机：模型与 adapter 都就位之后、optimizer 构造之前（optimizer_phase 在
    main() 里排在 models_phase 之后）。**参数对象身份不变**是这里的关键前提 ——
    ``nn.ParameterList`` 注册引用而不复制，DDP 构造也只是原地广播数值，所以
    optimizer 之后从 ``injector.get_param_groups()`` 拿到的仍是同一批张量，
    梯度同步与参数更新落在同一处内存上。

    构造参数逐条说明
    ----------------
    ``device_ids=[local_rank]`` / ``output_device=local_rank``：单设备模块的标准写法。
    ``distributed.init()`` 已经 set_device 过，这里再显式声明一次让 DDP 自己的
    输入搬运和 reduction 流都落在正确的卡上。

    ``find_unused_parameters``：**按 ``lora_module_dropout`` 决定**，见
    :func:`_needs_find_unused_parameters`。开着的代价是每步多一次 autograd 图全图
    遍历（PyTorch 文档称开销可观，真机上 DDP 也会主动警告「没找到未用参数，考虑
    关掉」）；关错了则会抛「Expected to have finished reduction in the prior
    iteration」。所以判据必须精确对应「本步是否真可能有参数不参与前向」。

    ``broadcast_buffers=False``：DiT 用 LayerNorm/RMSNorm，没有 BatchNorm 那种需要
    跨 rank 校正的 running stats，每步广播 buffer 纯属浪费带宽。更要紧的是
    T-LoRA 会按**本 rank 的 sigma_t** 每步写一份 rank mask buffer，广播 rank 0 的
    版本过去会直接算错别人的前向。（当前 shim 里只有 ParameterList、本来就没有
    buffer，显式写上是防以后往 shim 里加东西时踩坑。）

    ``static_graph``：保持默认 False。本循环的计算图逐步会变（T-LoRA 按 timestep
    改结构、module_dropout 改参与集合），static_graph 会把第一步的图当成永久事实。

    ``gradient_as_bucket_view``：仍然不开，但**原来写的理由是错的**，别照着它推理。

    原文说「只有 adapter 参数进桶（几十 MB 级），省下的量不值得」。前半句对，后半句
    只在低秩下成立：

        常规低秩 LoKr（rank 32 级，~20-50M 参数）  ->  76-191 MiB   原理由成立
        LyCORIS full matrix（803M 参数，fp32）      ->  2.99 GiB    差约 40 倍

    full matrix 是用户会刻意选的配置（``lora_dim`` 给个极大值触发），而它 OOM 时只差
    1.43 GiB —— 3 GiB 在这个处境下不是「不值得」，是决定性的。

    那为什么还是不开：**没验证过它和 ``set_to_none=True`` 的组合**。开了之后
    ``p.grad`` 是桶的视图，而本循环每步都调 ``ctx.optimizer.zero_grad()``（五处，
    全都没传 ``set_to_none``，torch>=2.0 默认就是 True），也就是每步都把 ``p.grad``
    置 None。reducer 是否在下一次反向前把视图重新指回去，取决于 C++ reducer 的实现
    细节，本机没装 torch、官方文档也取不到，无法核实。这个方向上猜错的后果是各 rank
    梯度静默不一致（不报错，只是训出来的东西不对），所以不靠推理开它。

    真要开，两件事必须一起做，缺一不可：
      1. 在真机上跑一步，确认 ``set_to_none=True`` + bucket view 在当前 torch 版本下
         各 rank 梯度仍然一致；
      2. 先补上 ``ppsf_fused_back_pass`` 的多卡拦截（见 bootstrap._check_ddp_compat）。
         它用 post-accumulate hook 在反向过程中就地更新参数并释放梯度，而 DDP 此时还
         要拿那块存储做 all_reduce —— 和 bucket view 是直接冲突。

    另外本循环对 ``.grad`` 只有「读」（第 691 行的有限性检查）和「原地改」
    (``clip_grad_norm_`` 的 ``mul_``)，没有任何一处给 ``p.grad`` 赋新张量 —— 那是
    bucket view 的必要条件，这一条是满足的，但不充分。
    """
    if not dist_env.is_distributed():
        return

    # DDP 初始化是集体操作（所有 rank 必须一起参与），但暂停信号可能只在某些
    # rank 上先到达，导致它提前退出、其他 rank 等死在 NCCL 初始化上。所以先做
    # 一个轻量同步：如果任何 rank 收到了暂停信号，全体一起退出（不进 DDP）。
    # 用 all_reduce 而非 barrier：barrier 在 process group 不存在时会失败，而
    # 暂停可能发生在 init_process_group 之前。
    import torch
    if ctx.pause_signal_seen:
        # 本 rank 已收到暂停 → 退出前通知对方。不用 all_reduce（它需要 pg），
        # 而是直接退 —— 对方也会在下面的检查点看到信号或在 DDP 初始化超时。
        logger.info("_wrap_ddp: 本 rank 已收到暂停信号，跳过 DDP 包装")
        raise KeyboardInterrupt("DDP 包装前检测到暂停信号")
    # 检查文件标记（loop.py 的另一条暂停通道）：早期退出能避免 DDP 卡住。
    pause_marker = ctx.output_dir / ".pause"
    if pause_marker.exists():
        logger.info("_wrap_ddp: 检测到暂停标记文件，跳过 DDP 包装")
        raise KeyboardInterrupt("DDP 包装前检测到暂停标记")

    trainable = ctx.injector.get_params()
    if not trainable:
        raise RuntimeError(
            "多卡训练需要至少一个可训练参数，但 adapter 没有产出任何 "
            "requires_grad=True 的参数。请检查 lora_type / lora_rank 配置。"
        )

    from torch.nn.parallel import DistributedDataParallel

    local_rank = dist_env.local_rank()
    ctx.ddp_model = DistributedDataParallel(
        _DDPAdapterSync(ctx.model, trainable),
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=_needs_find_unused_parameters(ctx.args),
        broadcast_buffers=False,
    )
    logger.info(
        "DDP 已就绪：同步 %d 个 adapter 参数张量（%.1fM），基座权重 frozen 不参与通信"
        "%s",
        len(trainable), sum(p.numel() for p in trainable) / 1e6,
        "，find_unused_parameters 已开（lora_module_dropout>0）"
        if _needs_find_unused_parameters(ctx.args) else "",
    )


def _defer_dit_for_text_cache(ctx: TrainingContext) -> bool:
    return (
        ctx.family.spec.text.strategy == "cached_varlen"
        and bool(getattr(ctx.args, "text_encoder_cache", True))
    )


def _validate_fp8_base(ctx: TrainingContext) -> None:
    """fp8 底模（fp8_base 训练）的组合校验——fail-fast 于任何大加载之前。

    探测只读 safetensors header（毫秒级），非 fp8 底模零开销直通。目前只有
    krea2 loader 接受 fp8 checkpoint（Anima loader 自行拒绝），但探测本身
    族无关。两条硬约束：

    - grad_checkpoint 必须开：fp8 的显存收益依赖重算段释放逐层 dequant 的
      临时权重；不开则 autograd 全量驻留 264 层 bf16 副本，占用反超 bf16。
    - DoRA 不支持：lycoris weight_decompose 初始化读底模权重数值（范数），
      fp8 直接 cast 缺 scale 校正，数值不正确（与推理/merge 拒绝口径一致）。
    """
    from training.families.krea2.loader import checkpoint_contains_fp8

    args = ctx.args
    if not checkpoint_contains_fp8(getattr(args, "transformer_path", "") or ""):
        return
    problems = []
    if not bool(getattr(args, "grad_checkpoint", True)):
        problems.append(
            "grad_checkpoint=false：fp8 底模的逐层 dequant 临时权重会被 "
            "autograd 全量驻留，显存占用反超 bf16。请开启梯度检查点。"
        )
    if bool(getattr(args, "lora_dora", False)):
        problems.append(
            "lora_dora=true：DoRA 初始化读取底模权重数值，fp8 存储下数值"
            "不正确。请关闭 DoRA 或改用 bf16 底模。"
        )
    if problems:
        raise RuntimeError(
            "fp8 底模与当前配置不兼容：\n- " + "\n- ".join(problems)
        )
    logger.info(msg("train.fp8_base_detected"))


def run(ctx: TrainingContext) -> None:
    """Resolve paths and load either the complete stack or the cache-first half."""
    from training.sysmem import check_load_budget, guard_enabled_from_env

    if ctx.family is None:
        ctx.family = resolve_family(ctx.args)
    _resolve_paths(ctx)
    _validate_fp8_base(ctx)

    if _defer_dit_for_text_cache(ctx):
        logger.info(msg("train.text_cache_order"))
        # 分段预算：本段只加载 VAE + TE（DiT 由 finish 段单独预算）。
        # 开关来自 设置 → 训练 → 训练参数（supervisor 经 env 注入，默认开）。
        check_load_budget(
            guard_enabled_from_env(),
            weight_paths=[getattr(ctx.args, "vae_path", ""),
                          getattr(ctx.args, "text_encoder_path", "")],
            stage="训练模型加载（VAE/文本编码器）",
            settings_hint="设置 → 训练 → 训练参数",
        )
        _load_vae(ctx)
        _load_text(ctx)
        return

    # Preserve the historical Anima order. Storage-free Krea2 deliberately keeps
    # the DiT resident while its text encoder is loaded for per-batch encoding.
    check_load_budget(
        guard_enabled_from_env(),
        weight_paths=[
            getattr(ctx.args, "transformer_path", ""),
            getattr(ctx.args, "vae_path", ""),
            getattr(ctx.args, "text_encoder_path", ""),
        ],
        stage="训练模型加载",
        vram_discount_ratio=_swap_vram_discount(ctx),
        settings_hint="设置 → 训练 → 训练参数",
    )
    _load_dit(ctx)
    _load_vae(ctx)
    _load_text(ctx)
    _inject_adapter(ctx)
    _wrap_ddp(ctx)
    _log_train_start_vram(ctx)


def finish(ctx: TrainingContext) -> None:
    """Load/inject a DiT deferred by cached text preparation; otherwise no-op."""
    if ctx.model is not None:
        return
    from training.sysmem import check_load_budget, guard_enabled_from_env

    logger.info(msg("train.text_cache_done_loading"))
    check_load_budget(
        guard_enabled_from_env(),
        weight_paths=[getattr(ctx.args, "transformer_path", "")],
        stage="训练模型加载（Transformer）",
        vram_discount_ratio=_swap_vram_discount(ctx),
        settings_hint="设置 → 训练 → 训练参数",
    )
    _load_dit(ctx)
    _inject_adapter(ctx)
    # cached_varlen 族的 DiT 推迟到这里才加载，DDP 也必须跟着推迟 —— 包的时候
    # adapter 参数必须已经存在（见 _wrap_ddp）。两条路径各调一次、互斥（run()
    # 走 defer 分支时提前 return），不会重复包。
    _wrap_ddp(ctx)
    _log_train_start_vram(ctx)


def _read_lora_family(path) -> str:
    """Read artifact family; legacy unmarked safetensors grandfather to Anima."""
    import json

    from safetensors import safe_open

    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            meta = handle.metadata() or {}
        args = json.loads(meta.get("ss_network_args") or "{}")
        return str(args.get("model_family") or "anima")
    except Exception:
        return "anima"
