#!/usr/bin/env python
"""Anima LoRA Trainer v2 — main() 编排入口。

本模块的实现层已按 ADR 0003 PR-A 拆到 runtime/training/ 子包：
  bootstrap / cli / observability / model_loading / models / text_encoding /
  state / dataset / sampling / timestep_sampling / noise / loss_weighting

顶部的 re-export 段保留 anima_train.X 访问路径，给 sister script
（anima_daemon / anima_generate / anima_reg_ai）和 tests/ 不变。新代码请
直接 `from training.X import Y`。

LoRA / LoKr 实现：见 utils.lycoris_adapter.AnimaLycorisAdapter（ADR 0001）。
"""

import os
import sys
from pathlib import Path

# 小显存优化：减少显存碎片，缓解 8GB 卡 LoKr full-matrix OOM。
# - 必须在 torch 链式 import 之前设置：torch 在 import 阶段就读 alloc conf 并缓存，
#   之后再改无效。**正因为如此，这里不能用 utils.accelerator.detect()**（它 import
#   torch）—— 只能靠设备节点这种 stdlib 级别的信号判断后端。
# - expandable_segments 的 CUDA backend 实现需要 PYTORCH_C10_DRIVER_API_SUPPORTED 宏，
#   PyTorch 的 c10/cuda/CMakeLists.txt 把该宏 gate 在 `if(NOT WIN32)`，因此 Windows wheel
#   不包含该 backend，运行时会 emit `TORCH_WARN_ONCE("expandable_segments not supported
#   on this platform")` 并强制 disable。为避免 Windows 用户看无用 warning，只在 Linux 设。
# - 海光 DCU（DTK / HIP build）上**不设**：真机实测（DTK 26.04 / torch 2.5.1）HIP
#   allocator 不支持这个选项，设了会 emit
#   `UserWarning: expandable_segments not supported on this platform`
#   （来自 c10/hip/HIPAllocatorConfig.h）然后强制 disable —— 也就是有噪音没收益。
#   注意 DTK 会把 `PYTORCH_CUDA_ALLOC_CONF` 也喂给 HIP allocator（HIP build 复用
#   CUDA 命名），所以两个变量都得跳过，只跳 HIP 那个不管用。
#   小显存碎片优化在 DCU 上的替代手段是 block swap（本项目已支持），而 DCU 卡普遍
#   显存大（BW1000 有 64GB），碎片压力本身也小得多。
# - setdefault 不覆盖用户已显式设置的值：想在 DTK 上强试可以自己 export。
if sys.platform.startswith("linux") and not os.path.exists("/dev/kfd"):
    # /dev/kfd = HSA kernel driver，ROCm / DTK 栈的标志。用设备节点而非
    # utils.accelerator.detect() 判断，是因为后者 import torch —— 而本段必须在
    # torch import 之前跑完（torch 在 import 阶段读 alloc conf 并缓存）。
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 脚本在 runtime/ 下按裸脚本启动（`python runtime/anima_train.py`）。
# 把仓库根 + runtime/ 注入 sys.path，让 `import utils.*` / `import train_monitor` /
# `import training.*` 等不需要改成包导入。
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "runtime"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# 统一日志 bootstrap（docs/design/logging-target-state.md）：与 studio / worker
# 同一 formatter；console 级别读 ANIMA_LOG_LEVEL（supervisor spawn 注入 DEBUG，
# 人手跑默认 INFO）；内含 Windows 控制台 UTF-8 兜底（cp936 下中文变 \uXXXX 的
# 老问题）。process 名 / trace_id 优先取 supervisor 注入的 env（与 worker 一致），
# 人手跑退回脚本名。sister script（daemon / generate / reg_ai）import 本模块后
# 再各自调一次，process 名不同则替换成自己的一套 handler。
from studio.infrastructure.logging import (  # noqa: E402
    PROCESS_ENV, TRACE_ENV, bind_trace_id, new_trace_id, setup_logging,
)

setup_logging(os.environ.get(PROCESS_ENV) or "anima_train", file=False, console=True)
bind_trace_id(os.environ.get(TRACE_ENV) or new_trace_id())


# ─── Re-exports for sister script / tests (ADR 0003 PR-A) ────────────────────
# 这些名字被 anima_daemon / anima_generate / anima_reg_ai (`import anima_train as _T`
# 然后 _T.X) 以及 tests/test_anima_train_migration.py 等直接读取。新代码请
# 直接 import 子模块，不要再依赖 anima_train 顶层。
from training.bootstrap import (  # noqa: E402
    apply_yaml_config,
    ensure_dependencies,
    init_progress,
    load_yaml_config,
)
from training.observability import (  # noqa: E402
    WandBMonitor,
    init_wandb_monitor,
    render_curve_panel,
    render_loss_curve,
)
from training.model_loading import (  # noqa: E402
    _load_safetensors_state_dict,
    _load_weights_best_effort,
    _pick_best_prefix_remap,
    _strip_prefixes,
    enable_xformers,
    find_diffusion_pipe_root,
    forward_with_optional_checkpoint,
    resolve_path_best_effort,
)
from training.families.anima.text_encoding import (  # noqa: E402
    _build_qwen_text_from_prompt,
    _parse_weighted_tag,
    encode_qwen,
    tokenize_t5_weighted,
)
from training.state import load_training_state, save_training_state  # noqa: E402
from training.model_loading import ensure_models_namespace  # noqa: E402
from training.vae import load_vae  # noqa: E402
from training.families import get_family, resolve_family  # noqa: E402  # 派发咽喉（D8'）
from training.families.anima.loader import (  # noqa: E402
    load_anima_model,
    load_text_encoders,
)
from training.families.anima.sampling import sample_image  # noqa: E402
from training.dataset import (  # noqa: E402
    BucketBatchSampler,
    BucketManager,
    CachedLatentDataset,
    ImageDataset,
    MergedDataset,
    RepeatDataset,
    collate_fn,
    collate_fn_cached,
)
from training.cli import (  # noqa: E402
    parse_args,
    prompt_for_args,
)
from training.timestep_sampling import sample_t  # noqa: E402
from training.noise import make_noise  # noqa: E402
from training.loss_weighting import compute_loss_weight  # noqa: E402


# ============================================================================
# 主函数
# ============================================================================

def main():
    """ADR 0003 PR-B：main() 现在只编排 phase。

    每个 phase 是个 `run(ctx)` 函数，按顺序 in-place mutate TrainingContext。
    具体实现在 runtime/training/phases/。
    """
    from training import phases
    from training.context import TrainingContext
    from training import loop
    from utils import distributed

    args = parse_args()
    ctx = TrainingContext(args=args)
    # 进程组的生命周期横跨全部 phase（bootstrap 里 init、各 phase 都可能通信），
    # 所以销毁只能收在这一层 —— finalize_phase 只在成功路径执行，训练中途抛异常
    # 就不会走到，那正是留下僵死 NCCL 通信器的情形：同一张卡上的旧通信器没释放，
    # 下一个训练任务起来时可能卡死在 init_process_group 上。
    #
    # destroy() 幂等且吞掉自身异常，所以放在 finally 里不会盖掉真正的训练异常
    # （单进程下它整个是 no-op，这段对不开多卡的用户零影响）。
    # 启动期的停止检查点。每个 phase 在真机上都是数十秒到分钟级（12.9B 权重加载、
    # VAE latent 缓存、文本编码器缓存），这段时间里用户没有干净的停止方式 —— 只能
    # 取消，而取消走 SIGTERM，torchrun 的 elastic agent 会抛 SignalException 加
    # 30 行 traceback，看着像崩溃（真机实测）。
    #
    # 放在 phase **之间**而不是 phase 内部：粒度够用（响应延迟 = 一个 phase），而且
    # 天然保证所有 rank 在同一位置检查 —— phase 边界是各 rank 必然都会经过的点，
    # 塞进 phase 内部就得逐个确认那里是不是所有 rank 都走到。
    #
    # 传 ctx.emit 让消息进 task log；bootstrap 之前 ctx 还没 emit 能力，所以第一个
    # 检查点在 bootstrap 之后。
    try:
        phases.bootstrap.run(ctx)
        distributed.exit_if_pause_requested(ctx.emit)
        phases.models.run(ctx)
        distributed.exit_if_pause_requested(ctx.emit)
        phases.dataset.run(ctx)
        distributed.exit_if_pause_requested(ctx.emit)
        phases.text_cache.run(ctx)
        distributed.exit_if_pause_requested(ctx.emit)
        phases.models.finish(ctx)
        distributed.exit_if_pause_requested(ctx.emit)
        phases.optimizer.run(ctx)
        phases.resume.run(ctx)
        loop.run(ctx)
        phases.finalize.run(ctx)
    finally:
        distributed.destroy()


if __name__ == "__main__":
    main()
