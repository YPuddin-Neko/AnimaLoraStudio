"""bootstrap_phase：args + yaml + 交互 + seed + device/dtype + 输出目录 + wandb + monitor_state。

抽自 main() L113-185（ADR 0003 PR-B）。
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import torch

from training.bootstrap import apply_yaml_config, ensure_dependencies, load_yaml_config
from training.cli import prompt_for_args
from training.context import TrainingContext
from training.observability import WandBMonitor, init_wandb_monitor

# 模块对象 import（不是 `from utils.distributed import ...`）——测试要 monkeypatch
# `utils.distributed.*`，绑成本模块全局名会打不中。同 training/dataset.py 的约定。
from utils import distributed as dist_env


logger = logging.getLogger(__name__)


def _maybe_apply_pause_snapshot(args, resume_state_path: Path) -> None:
    """读 pause snapshot 覆盖 args（ADR 0006 PR-3 / §5.7）。

    args.resume_state = `…/pause_step_<N>.pt` → snapshot = `…/pause_step_<N>.config.json`。
    snapshot 不存在 → 静默跳过（用户走 ResumeFieldPicker 选周期 save 文件
    起新 task 的旧路径）。

    覆盖规则：
    - snapshot["args"] 内所有字段写到 args namespace，**例外**：
      - `resume_state` 不覆盖（snapshot 记录的是 pause 前的 args，那时 resume_state
        是空；现在我们才用它续训）
      - `config` 不覆盖（snapshot 记录的是用户当时的 yaml 路径，用户可能已删/改名）
    - snapshot["sample_prompts"] → args.sample_prompts（resume_phase 会读这个）
    """
    snapshot_path = resume_state_path.with_suffix(".config.json")
    if not snapshot_path.exists():
        return  # 不是 pause state，沿用现有 args
    try:
        raw = snapshot_path.read_text(encoding="utf-8")
        snapshot = json.loads(raw)
    except Exception as exc:
        logger.warning(
            f"读取 pause snapshot 失败，沿用现有 args: {snapshot_path} ({exc})"
        )
        return
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("args"), dict):
        logger.warning(f"pause snapshot schema 不识别，沿用现有 args: {snapshot_path}")
        return
    logger.info(f"加载 pause snapshot 覆盖训练参数: {snapshot_path}")
    snap_args: dict = snapshot["args"]
    skipped = {"resume_state", "config"}
    for k, v in snap_args.items():
        if k in skipped:
            continue
        setattr(args, k, v)
    sp = snapshot.get("sample_prompts")
    if isinstance(sp, list):
        args.sample_prompts = sp


def _check_navit_prerequisites(args) -> None:
    """NaViT 打包需要**可用的** xformers —— 启动期 fail-fast，不留到 forward 才炸。

    NaViT 走块对角 varlen 注意力，实现依赖 xformers 的 ``BlockDiagonalMask``
    （``modeling/anima/cosmos_predict2_modeling.py`` 的 ``torch_attention_op``）。
    schema 侧已把 ``attention_backend`` 钉成 ``xformers``，但那只保证配置自洽，
    **不保证包真的装了且能调用**：

    - 没装 xformers → ``set_attention_backend("xformers")`` 静默返回 ``"none"``，
      模型照常加载，直到第一个 step 进 attention 才抛 RuntimeError。用户此时
      已经等完了权重加载 + latent 缓存，白等好几分钟。
    - 海光 DCU 上 xformers **根本装不了**（官方只发 CUDA wheel），所以 DCU 用户
      无论如何都要关掉 NaViT。给他们看「装 xformers」的建议是误导，要单独给文案。

    SDPA 不是可行替代：它只吃 dense 加性 mask，块对角在 SDPA 上要展开成
    ``[B, 1, S, S]`` 的 O(S²) 张量，正好抵消打包省下来的算力与显存
    （token_budget=16384 时单个 mask 就是几百 MB），且会把 SDPA 顶到 math 后端。
    """
    if not bool(getattr(args, "navit_packing", False)):
        return
    try:
        import xformers.ops  # noqa: F401
        return
    except Exception as exc:  # noqa: BLE001  未装 / ABI 不匹配 / import 期崩都算不可用
        reason = f"{type(exc).__name__}: {exc}"

    from utils.accelerator import detect

    info = detect()
    if info.backend == "dcu":
        # 注意措辞：DCU 上 xformers **不是装不上**（海光在光合社区发布配套 wheel，
        # 与 DTK / torch 版本严格配套），只是不能靠 pip 自动装。早期版本这里写的是
        # 「装不上、请关掉 navit」，会让本来能用的用户白白放弃功能。
        raise RuntimeError(
            f"navit_packing=True 需要 xformers 的块对角 varlen 内核，但当前 import 失败。\n"
            f"  当前后端 {info.vendor_label}：公开源上没有 DCU 版 xformers，需从光合开发者"
            f"社区取与镜像 DTK / torch 版本匹配的 wheel 手动 pip install\n"
            f"  （文件名形如 xformers-0.0.33+das.opt1.dtk2604.torch251-py3-none-any.whl）。\n"
            f"  不想装就关掉 navit_packing，改用 ARB 分桶路径（功能等价、速度略低）。\n"
            f"  底层原因：{reason}"
        )
    raise RuntimeError(
        f"navit_packing=True 需要 xformers，但当前 import 失败。\n"
        f"  请安装（设置 → 训练 → xformers 一键装），或关闭 navit_packing。\n"
        f"  底层原因：{reason}"
    )


def _check_ddp_prerequisites(args) -> None:
    """多卡（``world_size>1``）与部分单卡特性互斥 —— 启动期 fail-fast。

    放在这里而不是 models_phase：这些组合一旦成立就没救，而 models_phase 之前
    已经要读 26GB 权重 + 缓存 latent，跑到那里才炸等于让用户白等好几分钟
    （与 ``_validate_fp8_base`` 的「fail-fast 于任何大加载之前」同一理由）。

    单进程下整个函数是 no-op —— 下面每条都以 ``is_distributed()`` 为前提。

    刻意**不**拦的两项（曾以为要拦，实测可行，记下来免得后人重复纠结）：

    - ``grad_checkpoint``：Anima 的检查点前向手工展开了模型内部
      （``prepare_embedded_sequence`` / ``blocks`` / ``final_layer``），压根不走
      ``model.__call__``。这本来会让 DDP 完全失效，但我们包的是 adapter 参数而不是
      DiT（见 models_phase._DDPAdapterSync），前向由 shim 转调，展开与否都能同步。
      两个族的 checkpoint 都是 ``use_reentrant=False``（reentrant 版才与 DDP 冲突）。
    - ``navit_packing``：同理，走 ``model.forward_packed_navit`` 也经 shim。
    """
    if not dist_env.is_distributed():
        return

    problems: list[str] = []

    # 1) block swap：把末尾若干层留在 CPU pinned，前向时逐块换入换出。DDP 在构造时
    #    按「参数常驻单一设备」建梯度桶并注册 hook，参数在 CPU/GPU 之间跳会让桶与
    #    设备假设同时失效（行为未定义，最可能是构造期报设备不一致或反向卡死）。
    #    不静默关掉任何一方：用户开 block swap 是为了塞进小显存，开多卡是为了摊
    #    算力，猜错方向都会让人以为「设置生效了」。
    blocks_to_swap = int(getattr(args, "blocks_to_swap", 0) or 0)
    if blocks_to_swap > 0:
        problems.append(
            f"blocks_to_swap={blocks_to_swap}（block swap）与多卡同时开启。"
            f"DDP 假设参数常驻单一设备，block swap 会把部分层挪到 CPU，两者行为未定义。"
            f"请二选一：把 blocks_to_swap 置 0 跑多卡（多卡本身就是为了摊显存/算力），"
            f"或改用单卡 + block swap（去掉 torchrun，直接 python runtime/anima_train.py）。"
        )

    # 2) leap / FlowBP：一个 micro-batch 里要跑 2~6 次前向再合成一个 loss。DDP 每次
    #    forward 都会 arm 一次 reducer，第二次 arm 时上一次的 reduction 还没 finalize
    #    → 直接抛「Expected to have finished reduction in the prior iteration」。
    #    这是 DDP 的硬约束（一次 backward 对一次 forward），不是调参能绕过的。
    if bool(getattr(args, "leap_enabled", False)):
        problems.append(
            "leap_enabled=true 与多卡同时开启。LeapAlign/FlowBP 每步要跑多次前向再合成"
            "一个 loss，而 DDP 要求「一次 forward 对一次 backward」，第二次前向就会抛"
            "「Expected to have finished reduction in the prior iteration」。"
            "请关闭 leap_enabled 跑多卡，或用单卡跑 leap。"
        )

    # 3) SRA v2 + 梯度裁剪：SRA 的 projection MLP 在 optimizer 里，但它的 loss 在
    #    DDP 前向之外算（loop.py 拿 hook 抓的激活），梯度进不了 DDP 的 reduction ——
    #    也不能硬塞进去（find_unused_parameters 会先把它判成未用、提前 mark ready，
    #    真梯度随后才到，等于「同一变量 mark 两次」）。
    #    MLP 各 rank 独立本身无害（训练完就丢，见 finalize），但
    #    clip_grad_norm_ 是对 **ctx.trainable_params 全体** 求范数：未同步的 SRA 梯度
    #    混进去 → 各 rank 的 clip 系数不同 → 同一份已同步的 LoRA 梯度被乘上不同倍数
    #    → 权重从此分歧。这个是静默的（不报错、只是训出来的东西不对），所以要拦。
    if bool(getattr(args, "sra_enabled", False)) and float(getattr(args, "grad_clip_max_norm", 0) or 0) > 0:
        problems.append(
            "sra_enabled=true 与 grad_clip_max_norm>0 在多卡下不能同用：SRA projection MLP "
            "的梯度不参与 all_reduce，混进全局梯度范数会让各 rank 算出不同的裁剪系数，"
            "已同步的 LoRA 梯度被乘上不同倍数 → 权重静默分歧。"
            "请把 grad_clip_max_norm 置 0，或关闭 sra_enabled。"
        )

    if problems:
        raise RuntimeError(
            f"多卡训练（world_size={dist_env.world_size()}）与当前配置不兼容：\n- "
            + "\n- ".join(problems)
        )

    # 拦不住但要提醒：SRA MLP 在多卡下退化为 per-rank 独立探针。
    if bool(getattr(args, "sra_enabled", False)):
        logger.warning(
            "多卡 + SRA v2：projection MLP 的梯度不跨 rank 同步，每个 rank 各训一份"
            "（训练结束即丢弃，不进 LoRA 产物）。经它流回 LoRA 的那部分梯度仍由 DDP "
            "正常同步，故 LoRA 权重不受影响。"
        )


def _resolve_ddp_device(base_device: str) -> str:
    """多卡下把裸 ``"cuda"`` 收窄成 ``"cuda:<local_rank>"``；其余情况原值返回。

    为什么必须收窄：``ctx.device`` 会被当成显式目标传给 ``.to(device)`` /
    ``torch.zeros(device=...)`` 等几十处。裸 ``"cuda"`` 解析成「当前设备」，而
    ``distributed.init()`` 已经 ``set_device(local_rank)``，多数情况下确实指对了 ——
    但只要有任何一处在别的线程/流里（dataloader worker、采样、VAE 分块）没继承到
    当前设备，张量就会悄悄落到 0 号卡：表现是 0 号卡显存翻 N 倍然后 OOM，而其余卡
    闲着。写死 index 就没有这个歧义。

    为什么单卡**必须**保持裸 ``"cuda"`` 而不是 ``"cuda:0"``：那是
    ``accelerator.detect().torch_device`` 的原值，既有行为（含存进 checkpoint 的
    device 字符串、日志、以及测试里对 "cuda" 的字面断言）都按它来。单卡改成
    "cuda:0" 是无谓的行为变更。

    非 CUDA 后端（CPU）不动：``"cpu:0"`` 虽然合法但没有意义，且 gloo 调试路径下
    world_size>1 也不该改设备。
    """
    if not dist_env.is_distributed():
        return base_device
    if base_device != "cuda":
        return base_device
    return f"cuda:{dist_env.local_rank()}"


def _configure_rank_logging() -> None:
    """多卡下给日志加 rank 前缀，并把非 rank 0 的 INFO 压掉。

    两件事一起做才有意义：
    - **压 INFO**：N 个 rank 各打一份「加载 Transformer…」「数据集大小…」，task log
      直接 N 倍冗余，真正的问题被淹掉。
    - **保留 WARNING/ERROR**：非 0 rank 的崩溃原因只在它自己的 stderr 里，压掉等于
      多卡出事永远查不到根因（只看得到 rank 0 说「进程组挂了」）。所以门槛设在
      WARNING 而不是干脆关掉 handler。
    - **加前缀**：所有 rank 写同一个 stdout，交错的 WARNING 不标 rank 等于没法归因。
      rank 0 也加 —— 只有多卡时才加，单进程日志格式逐字节不变。

    改 root logger 而不是逐个 module logger：业务代码的 logger 全部 propagate 到
    root（``anima_train.py`` 顶部一次 basicConfig），在 root 上收敛是唯一能覆盖到
    「本次改动碰不到的文件」的做法。
    """
    if not dist_env.is_distributed():
        return
    root = logging.getLogger()
    prefix = f"[rank {dist_env.rank()}] "
    for handler in root.handlers:
        handler.setFormatter(
            logging.Formatter(f"%(asctime)s - %(levelname)s - {prefix}%(message)s")
        )
    if not dist_env.is_main():
        root.setLevel(logging.WARNING)


def _resolve_sample_seed(args) -> None:
    """sample_seed=0 → 训练开始时随机抽一次写回 args，并 log。

    Why：sample_seed=0 走 sample_runner 时不调 torch.manual_seed，整批
    采样跟着 global RNG 漂移，跨 epoch 同 prompt 出图不同 → 看不出是
    模型收敛还是噪声变了。抽一次固定下来，整轮训练同 prompt 同 seed。

    与 pause snapshot 协作：snapshot 写整份 args.dict()，resolved 值会
    被 freeze；resume 经 _maybe_apply_pause_snapshot 灌回，跨 pause 仍
    用同一 seed。用户重新起 task 时若 yaml 还是 0，启动重抽一次新随机。
    """
    if int(getattr(args, "sample_seed", 0) or 0):
        return
    args.sample_seed = random.randint(1, 2**31 - 1)
    logger.info(f"sample_seed=0 → 训练用随机种子: {args.sample_seed}")


def run(ctx: TrainingContext) -> None:
    """完成训练前一切非模型/数据的准备：

    - 加载 yaml config（如有）+ 交互模式补缺字段
    - ensure_dependencies
    - 设种子 / 选 device / dtype
    - 建 output_dir + sample_dir
    - 初始化 wandb_monitor + monitor_state.json 写入器
    """
    args = ctx.args

    # PR-C：启动期校验所有 plugin 子包 schema 一致性，避免运行半天才发现配错
    from training.adapters import validate_schema_consistency as _validate_adapters
    from training.losses import validate_schema_consistency as _validate_losses
    from training.optimizers import validate_schema_consistency as _validate_optimizers
    from training.schedulers import validate_schema_consistency as _validate_schedulers
    _validate_adapters()
    _validate_optimizers()
    _validate_schedulers()
    _validate_losses()

    # 加载 YAML 配置文件 + TrainingConfig 归一（刀 1 / R1）。无 yaml 的纯 CLI
    # 路径同样要走：parse_args 的 sparse namespace 缺 schema 默认值，由
    # apply_yaml_config 经 pydantic 构造统一补齐（迁移 / 族 overlay / 校验一并生效）
    config = {}
    if args.config:
        logger.info(f"加载配置文件: {args.config}")
        ctx.config_path = Path(args.config).resolve()
        ctx.config_dir = ctx.config_path.parent
        config = load_yaml_config(args.config)
    ctx.args = apply_yaml_config(args, config)
    args = ctx.args

    # bridge 已为 prefer_json bool 自动产生 --prefer-json / --no-prefer-json，
    # 此处无需再做兼容处理。

    # ADR 0006 PR-3：pause 文件旁边的 .config.json snapshot 覆盖 args。
    # 触发条件：args.resume_state 指向的 .pt 旁边有同前缀的 .config.json。
    # 仅 pause 触发的 state 会带 snapshot（PR-2 handle_interrupt 写）；周期
    # save 没有 snapshot，ResumeFieldPicker 起新 task 走原路径（用户当前
    # yaml config）。Snapshot freeze 是 ADR §5.7 的核心 — resume 时 task 的
    # 训练参数严格用暂停那一刻的值，跟用户后续改 version / preset / yaml
    # 完全解耦。
    if getattr(args, "resume_state", None):
        _maybe_apply_pause_snapshot(args, Path(args.resume_state))
        ctx.args = args

    # 交互模式检查
    required = [args.data_dir, args.transformer_path, args.vae_path, args.text_encoder_path]
    if args.interactive or any(not x for x in required):
        ctx.args = prompt_for_args(args)
        args = ctx.args

    # 多模型 PR-2b：族解析 fail-fast（args 定稿后、任何权重加载前；未知
    # model_family 即死。pause snapshot 已 freeze args → 跨 pause 族一致性免费）
    # 能力校验不再单独做（刀 1 / R1）：apply_yaml_config 的 TrainingConfig
    # 构造已跑 _validate_family_capabilities，CLI 直达路径与 Studio 同一防线。
    from training.families import resolve_family

    ctx.family = resolve_family(args)

    # 审计 #2（设计文档 §10.1）：T-LoRA rank mask 按 batch 均值 timestep 生成，
    # batch>1 时 per-sample「高噪声低 rank」退化为批均值近似 —— 不拦（硬拦会
    # 误伤想跑小 batch 的用户），启动期显式提示
    if getattr(args, "lora_type", "") == "tlora" and int(getattr(args, "batch_size", 1)) > 1:
        logger.warning(
            "T-LoRA 与 batch_size=%s 同用：rank mask 按 batch 均值 timestep 生成，"
            "per-sample 掩码退化为批均值近似；要获得论文行为请用 batch_size=1",
            args.batch_size,
        )

    ctx.args = args

    # 依赖检测
    ensure_dependencies(auto_install=args.auto_install)

    # 延迟导入：保留原 main() 顺序 —— ensure_dependencies 之后才能 import numpy/PIL
    import numpy as np

    # 设置随机种子。
    #
    # 多卡下 **torch / numpy 偏移 rank，Python random 不偏移** —— 这个不对称是故意的：
    #   - torch 管的是「数据级随机」：噪声（make_noise）、timestep 采样。所有 rank
    #     用同一种子时它们会抽出**完全相同**的 t 向量和噪声张量，world_size=4 时
    #     全局每步只覆盖 4 个 timestep 而不是 16 个 —— flow matching 下这直接削弱
    #     每步的噪声水平覆盖，是白扔算力。
    #   - Python ``random`` 管的是「控制流级随机」：loop.py 的 leap 掷骰子
    #     （走 leap 目标还是传统目标）。这类分支必须各 rank 一致，否则计算图不同构、
    #     梯度桶对不上。偏移它就是自找死锁。
    # 分片本身不受影响：BucketBatchSampler 用自带的 ``random.Random(seed + epoch)``
    # 实例算 batch 顺序（dataset.py），与全局 RNG 无关 —— 所以各 rank 仍生成同一份
    # 完整 batch 列表再 ``[rank::ws]`` 切，不会重叠或漏采。
    _seed_offset = dist_env.rank()
    torch.manual_seed(args.seed + _seed_offset)
    random.seed(args.seed)
    np.random.seed(args.seed + _seed_offset)

    _resolve_sample_seed(args)
    _check_navit_prerequisites(args)
    _check_ddp_prerequisites(args)

    # 进程组初始化 + 绑卡。**必须在 ctx.device 定下来之前**：init() 内部要
    # torch.cuda.set_device(local_rank)，而下面 _resolve_ddp_device 要读绑定结果。
    # 单进程下 init() 是 no-op 并返回 False，所以无条件调。
    dist_env.init()
    _configure_rank_logging()
    if dist_env.is_distributed():
        logger.info("训练拓扑：%s", dist_env.topology_summary())
    # 销毁进程组的**兜底**注册点。finalize_phase 会在正常收尾时显式 destroy()，
    # 但异常退出（未捕获异常冒出 main）和 handle_interrupt 的 sys.exit(0) 都不会
    # 走到那儿 —— 而不销毁会留下僵死 NCCL 通信器，下一个训练任务可能卡在
    # init_process_group 上（同卡旧通信器没释放）。atexit 是唯一能同时覆盖这三条
    # 路径的挂点（main() 不在本次改动范围内，包不了 try/finally）。
    # destroy() 幂等且失败不抛，重复调用安全。
    import atexit

    atexit.register(dist_env.destroy)

    # HIP build（海光 DCU）上同样是 "cuda" —— ROCm 把 CUDA 设备 API 整套复用，
    # 写 "hip" 会 RuntimeError。判定收敛在 utils.accelerator（单一权威源），
    # 这里不要改回自己读 torch.version.*。
    from utils.accelerator import configure_sdpa, detect as _detect_accelerator

    # 多卡时收窄成 cuda:<local_rank>；单卡保持裸 "cuda"（见 _resolve_ddp_device）。
    ctx.device = _resolve_ddp_device(_detect_accelerator().torch_device)

    # 关掉本机实测不可用的 SDPA 后端。**必须在任何 forward 之前**，且必须在
    # ctx.device 定好之后（探测要真跑一次小 SDPA，需要设备就绪）。
    #
    # 为什么训练启动期就得做：海光 DTK 的 torch 开着 flash 后端但 kernel 在外部
    # flash_attn_2_cuda*.so 里，包没装时 SDPA 的默认 dispatch 会先试 flash、抛
    # RuntimeError 且**不回落** —— 也就是第一个 attention 就崩。见 configure_sdpa()。
    # NVIDIA 上本调用只做探测、不改任何开关。
    configure_sdpa()
    if args.mixed_precision == "bf16":
        ctx.dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        ctx.dtype = torch.float16
        ctx.scaler = torch.cuda.amp.GradScaler()
    else:
        ctx.dtype = torch.float32
    # VAE 精度与训练精度解耦：fp16 路径下 VAE 仍用 fp32（见 TrainingContext.vae_dtype）；
    # bf16/fp32 时 VAE 跟随主精度不变。
    ctx.vae_dtype = torch.float32 if ctx.dtype == torch.float16 else ctx.dtype

    # 创建输出目录
    ctx.output_dir = Path(args.output_dir)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    # 采样图落到 task 档案根的 samples/。supervisor 按 task 注入
    # `--monitor-state-file <studio_data>/tasks/<id>/monitor/state.json`，
    # sample_dir 取其上跳一层的 `samples/` —— `tasks/<id>/samples/`，跟 monitor/
    # 同级，整组（snapshot/ monitor/ samples/ run.log）就是 task 完整档案。
    # 没传 --monitor-state-file（纯 CLI 训练 / 兼容老版本注入路径）退回
    # output_dir/samples，samples.py 仍可在 monitor_dir 周围多候选搜回。
    # ── 非 rank 0 的「对外输出」全部关掉（多卡专用；单进程恒不进这个分支）──
    #
    # 收在这一处而不是逐个调用点，是因为几个关键调用方（resume_phase 的启动基线
    # 采样、init_progress 的 Rich Live）不在本次改动范围内，只能靠改它们读的 args /
    # ctx 字段来门控。三件事各有各的坑：
    #   - 进度条：N 个 Rich Live 抢同一个终端 → 互相覆盖出乱码。
    #   - 采样出图：所有 rank 会往**同一个** sample_dir/step_N.png 写，并发写同一
    #     文件必然写坏；而且采样是纯推理，多跑 N-1 份纯属浪费显存和时间。
    #
    # ⚠️ 置 0 之前**必须**先把周期值存进 ctx.sample_*_all_ranks。采样调用点外面套着
    # 一对 barrier（集合操作，必须所有 rank 都执行），它们的门控只能读那份 rank
    # 不变副本；读被改过的 args 会让非 rank 0 连 barrier 一起跳过 —— 真机上的后果是
    # NCCL 集合操作永久错位 + rank 1 OOM，详见 TrainingContext 上那两个字段的注释。
    #   - monitor_state.json：Studio 前端按 task 读一个文件，多 rank 同写会让
    #     step/loss 反复跳变。monitor_server=None 让 loop/resume 里所有
    #     `if ctx.monitor_server:` 分支自然短路。
    # args 被改的字段会进 auto_epoch_state.config.json（pause snapshot），但那份
    # snapshot 只由 rank 0 写，记录的是 rank 0 的真实值，resume 不受影响。
    # 先存 rank 不变副本（所有 rank 都执行这两行，值必然一致），再做 rank 相关抑制。
    ctx.sample_steps_all_ranks = int(getattr(args, "sample_steps", 0) or 0)
    ctx.sample_every_all_ranks = int(getattr(args, "sample_every", 0) or 0)
    if not dist_env.is_main():
        args.no_progress = True
        args.sample_steps = 0
        args.sample_every = 0

    _msf = getattr(args, "monitor_state_file", None)
    ctx.task_archive_dir = Path(_msf).parent.parent if _msf else None
    ctx.sample_dir = (ctx.task_archive_dir / "samples") if ctx.task_archive_dir else (ctx.output_dir / "samples")
    ctx.sample_dir.mkdir(parents=True, exist_ok=True)
    # ADR 0006 Addendum 2：auto_epoch_state.pt 同样归 task 档案 —— tasks/<id>/state/，
    # 跟 samples/ 同根。没传 --monitor-state-file（纯 CLI）→ None，
    # ctx.auto_state_dir() fallback 到 output_dir/state/task_<id>/（行为不变）。
    ctx.task_archive_state_dir = (ctx.task_archive_dir / "state") if ctx.task_archive_dir else None
    # supervisor 启动训练时通过 env LORA_TASK_ID 注入 queue task id（ADR 0006）。
    # 用于 ctx.state_dir() 计算 per-task state 子目录；env 不存在时 fallback unknown。
    _env_tid = os.environ.get("LORA_TASK_ID")
    if _env_tid:
        try:
            ctx.lora_task_id = int(_env_tid)
        except ValueError:
            logger.warning(f"LORA_TASK_ID={_env_tid!r} 不是 int，按 unknown 处理")
    # wandb 只由 rank 0 上报。非 rank 0 给一个 run=None 的 WandBMonitor：它的
    # `enabled` 属性为 False，log / log_image / upload_* / finish 全部提前 return ——
    # 也就是调用方（loop.py 每步都无条件 `ctx.wandb_monitor.log(...)`）不需要任何
    # 门控。比在几十个调用点加 if 可靠得多，也不会 AttributeError。
    # 为什么不是让每个 rank 各建一个 run：那会在 wandb 上多出 N-1 条曲线完全相同、
    # 但 step 对不齐的 run，面板直接没法看。
    if dist_env.is_main():
        ctx.wandb_monitor = init_wandb_monitor(args, ctx.output_dir, ctx.config_path)
    else:
        ctx.wandb_monitor = WandBMonitor(None, None)

    # Loss 函数（mse / huber；通过 losses/ plugin registry 派发）
    # 不依赖 total_steps，跟 timestep_sampler/scheduler 不同；放 bootstrap 而非
    # optimizer phase 避免架构错位。
    from training.losses import build_loss
    ctx.loss_fn = build_loss(args)

    # 训练监控状态写入（PP6.1）：永远开启，文件路径优先来自 --monitor-state-file，
    # 否则落到 output_dir/monitor_state.json。Studio 前端通过 /api/state?task_id=
    # 读这个文件，不再启动训练侧 HTTP server（Studio 自己是 monitor）。
    # 多卡：只有 rank 0 写状态文件。None 让 loop / resume 里所有
    # `if ctx.monitor_server:` 分支短路（含 save_training_state 时的 get_state()）。
    # 不 set_state_file 也保证 train_monitor 模块级的路径保持 None，
    # 即便有漏掉的调用点也不会真落盘。
    # （这是 run() 的最后一段，early return 即 phase 结束。）
    if not dist_env.is_main():
        ctx.monitor_server = None
        return

    ctx.monitor_server = True  # 兼容下方分支判断；实际代表「写状态文件」
    try:
        from train_monitor import set_state_file, update_monitor
        state_path = (
            Path(args.monitor_state_file)
            if getattr(args, "monitor_state_file", None)
            else ctx.output_dir / "monitor_state.json"
        )
        set_state_file(state_path)
        update_monitor(
            total_epochs=int(args.epochs or 0),
            config={
                "model": {"lokr": "Anima LoKr"}.get(args.lora_type, "Anima LoRA"),
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.learning_rate,
                "resolution": args.resolution,
                "data_dir": str(args.data_dir),
            },
        )
        logger.info(f"📊 训练监控状态文件: {state_path}")
    except Exception as e:
        logger.warning(f"监控状态写入初始化失败: {e}")
        ctx.monitor_server = None
