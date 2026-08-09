# 0016 — 双加速器后端：NVIDIA CUDA 与海光 DCU 并存

**状态**：Accepted
**日期**：2026-08-10
**决策者**：项目维护者

## 背景

此前本项目只支持 NVIDIA：README 硬件要求写明「A 卡 / Apple Silicon 不支持」，
装包链路（`tools/select_torch_index.py`、`studio/services/runtime/torch.py`）按
`nvidia-smi` 报的驱动版本选 `download.pytorch.org/whl/cuXXX` wheel，显存监控直连
NVML，flash-attn 走 GitHub 上按 `cuXXX`+`torchX.Y`+`cpXXX` 命名的 prebuilt wheel。

需求是在海光 DCU（Hygon DCU，型号 BW1000）上跑训练，环境为厂商镜像
`pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.11`。海光 DTK 是 ROCm 的分支，其 PyTorch
把 CUDA API 映射到 HIP。

摸清耦合点后，事实分成两类，二者的处理代价差了一个数量级：

**训练主链路几乎不需要改。** `torch.cuda.*`（66 处 `is_available`、37 处
`empty_cache`、28 处 `synchronize`、Stream / Event / pinned memory /
`mem_get_info` / `OutOfMemoryError`）在 HIP build 上语义相同；`torch.device("cuda")`
在 DTK 上同样是正确写法（写 `"hip"` 反而 `RuntimeError`）；`torch.autocast("cuda")`
与 `torch.cuda.amp.GradScaler` 照常工作。项目也没有自定义 CUDA kernel、没设
`allow_tf32` / `cudnn.benchmark`、fp8 走「存储 fp8 + 前向 dequant」而非原生
`_scaled_mm`（`quant_fp8.py` 为对齐 ComfyUI 逐位一致刻意如此），因此不依赖任何
NVIDIA 专有算子。

**装包与检测层会误判，且其中一条误判是破坏性的。** DTK wheel 上
`torch.version.cuda is None`、`torch.version.hip` 有值、版本串形如
`2.9.0+das.dtk2604`（既无 `+cu` 也无 `+cpu` 后缀）。原代码用
`torch.version.cuda is None` 判定「CPU-only 误装」，于是在 DCU 上：启动期弹「检测到
GPU 但装了 CPU 版 PyTorch」大警告、flash-attn 安装被 pre-check 拒绝、Settings 推荐
「一键重装为 cu128」。

最后一项是真正的危险：**DTK torch 由厂商镜像预装、不在 PyPI 上、与镜像内 DTK 运行时
严格配套**。任何 `pip install torch` 都会把它替换成 PyPI 的 CPU 版或 NVIDIA 版，
环境当场报废且**无法用 pip 装回来**——用户只能重建容器。同源风险还有两处：
`requirements.txt` 里裸的 `torch>=2.0.0` / `torchvision>=0.15.0`，以及训练侧
`--auto-install` 会 `pip install torchvision`（连带拉 CPU 版 torch）。

## 候选方案

1. **替换后端**：把 `nvidia-smi` / cu wheel / NVML 那套直接换成 DCU 版。改动最小、
   代码最直，但仓库从此不能在 NVIDIA 上跑，也没法合回上游。
2. **每个调用点就地 if**：在需要的地方各写 `torch.version.hip is not None` 判断。
   不新增抽象，但判定逻辑会散进十几个文件，后续加第三个后端时要全量重扫；也违反
   项目的单一权威源约定（`docs/AGENTS.md` §3.3）。
3. **抽一层后端识别 + 能力查询**：新增 `utils/accelerator.py` 作为唯一权威源，上层
   全部改成问它；训练主链路不动（因为本来就兼容）。

## 决策

采用方案 3。

**识别层**：新增 `utils/accelerator.py`，位于依赖链最底层（`docs/AGENTS.md` §3.1
的 `utils/` → `runtime/` → `studio/` → `tools/`），`runtime/` 与 `studio/` 都可
import，它自己不反向 import。torch 是它的**可选**依赖——`probe_stdlib()` 纯 stdlib，
供 venv 首装（只有 pip）时的 bootstrap 阶段使用。

- 后端标识 `Backend = Literal["cuda", "dcu", "cpu"]`。刻意不叫 `rocm`：只对海光 DCU
  做过验证，AMD Radeon / Instinct 虽同为 HIP build 但驱动栈与 smi 工具不同，不承诺
  未验证的支持。
- `detect()` 按 `torch.version.hip` 优先、`torch.version.cuda` 次之判定，结果进程内
  缓存（torch build 在进程生命周期内不变）。
- 能力查询而非后端判断作为上层接口：`should_manage_torch_install()`、
  `supports_prebuilt_flash_attn_wheels()`、`supports_xformers()`、
  `onnx_gpu_provider()`。上层问「能不能做这件事」，不问「你是什么卡」——加第四个后端时
  改这一处即可。
- `AcceleratorInfo.torch_device` 在 DCU 上返回 `"cuda"`。这个 property 的价值是让调用
  方不必自己判断，而不是真会返回第三个值。

**护栏层**（本次移植的核心，按失败代价排序）：

- `should_manage_torch_install()` 在 DCU 上恒 False。`reinstall()` 抛 RuntimeError、
  CLI `--torch=<tag>` 拒绝、Settings 重装按钮置灰。
- `studio.sh` 首装用 `select_torch_index.py --backend` 区分「DCU」与「无驱动」——原来
  两者都表现为静默无输出，而 caller 的正确动作恰好相反（DCU 必须完全不碰 torch；无驱动
  则装 PyPI 默认）。装 `requirements.txt` 时钉住 torch / torchvision 不被 pip 动。
- `ensure_dependencies(auto_install=True)` 在 DCU 上拒绝自动装 torch 生态包，fail-fast
  并指引从 DTK 渠道获取。只在 DCU 上 gate，NVIDIA 路径行为不变。

**能力边界**（DCU 上不可用，明确 fail-fast 而非静默降级）：

- **xformers**：公开源（PyPI / PyTorch index）只有 CUDA build，**自动安装**在 DCU 上
  拒绝。但海光在光合开发者社区发布配套 wheel（如
  `xformers-0.0.33+das.opt1.dtk2604.torch251`），手动装上后功能可用。

  这一条是设计中途修正过的错误假设：最初把「DCU 不支持 xformers」写进能力矩阵，导致
  `supports_xformers()` 恒 False、NaViT 打包在 DCU 上 fail-fast 并告诉用户「装不上」。
  实际上能力查询混淆了两件事——**能不能 pip 自动装**（DCU 上不能）与**装了能不能用**
  （能）。现已拆成 `can_pip_install_xformers()`（按后端）与 `xformers_works()`（实测
  import + 真调一次 `memory_efficient_attention`）。

  实测而非只 import 的理由：海光的 xformers wheel 是 `py3-none-any`（纯 Python），
  底层 kernel 依赖 DTK 侧实现，import 成功不代表能算。

  教训记在这里：**能力矩阵的条目要么是「该后端物理上不可能」，要么就得实测**。凭
  「上游没发 wheel」推断「这个后端不支持」，会漏掉厂商自建供应链这种情况。

**SDPA 后端必须实测，不能查标志位**（真机验证后追加的决策）。BW1000 / gfx936 /
DTK 26.04 实测，装 flash-attn 前后对比：

```
                    未装 flash-attn (torch 2.9.0)        装了 (torch 2.5.1 + flash-attn 2.6.1)
sdpa_default        RuntimeError: No matching             OK
                      libraries ... flash_attn_2_cuda*.so
sdpa_flash          同上                                  OK
sdpa_mem_efficient  RuntimeError: No available kernel     RuntimeError: No available kernel
sdpa_math           OK                                    OK
```

DTK 的 torch 编译时**开启**了 flash 后端，但把 kernel 委托给外部
`flash_attn_2_cuda*.so`（海光把 flash-attn 作为独立包发布，不在 PyPI）。包没装时
`torch.backends.cuda.flash_sdp_enabled()` 仍报 True，真正调用才抛异常——**所以静态
查标志位判断不出来**。

关键点是包没装时 `sdpa_default` 也失败：SDPA 的默认 dispatch 会先试 flash，撞上异常
直接抛出、**不会回落到 math**。这意味着「什么都不配置、依赖 SDPA 自己选」在缺包的 DCU
上等于训练第一个 attention 调用就崩。

`mem_efficient` 在两种情况下都不可用（DTK 编译时未开该后端），这是该平台的**正常终态**，
不是故障：flash 可用时它本就不会被选中。所以 `configure_sdpa()` 分级报告——flash 缺失
才 warn（附装包指引），只缺 mem_efficient 走 debug。

决策：`utils/accelerator.py:configure_sdpa()` 在训练启动期实测各后端（4 次小张量 SDPA，
约 1MB 显存），把失败的用 `torch.backends.cuda.enable_*_sdp(False)` 关掉，让 dispatch
不再去试。**只在 DCU 上改开关**，NVIDIA 侧仅探测不动。

选全局开关而不是在调用点包 `sdpa_kernel(MATH)`：调用点散落在 modeling 各处
（cosmos / krea2 / VAE / text encoder），全局开关一次覆盖；且用户之后装上海光 flash-attn
时探测会自动放行，无需改代码或配置。例外是 `comfy_qwen.py`——它用
`sdpa_kernel(priority, set_priority=True)`，那种写法会**覆盖**全局开关，所以改为从
`usable_sdpa_backends()` 取实测可用列表。

代价：只剩 math 时 attention 显式 materialize 完整 `[B, H, S, S]` 矩阵，比 flash 慢且
长序列吃显存。这是「能跑」与「跑得快」之间的取舍，装 flash-attn 即恢复。
- **NaViT 打包**：硬依赖 xformers 的 `BlockDiagonalMask` varlen 内核。DCU 上装了海光配套
  xformers 即可用；没装则训练启动期 fail-fast（原实现在第一个 step 进 attention 才抛，
  用户已白等完权重加载 + latent 缓存）。SDPA 不是可行替代：块对角在 SDPA 上要展开成
  `[B, 1, S, S]` 的 O(S²) dense mask，正好抵消打包省下的算力与显存，并把 SDPA 顶到
  math 后端。这条护栏做成**后端无关**：NVIDIA 用户没装 xformers 时同样受益。
- **flash-attn**：GitHub 那套 prebuilt wheel 是 CUDA-only。海光的 flash-attn 走 DTK 自家
  渠道，装好后本项目照常识别（训练侧只 `from flash_attn import flash_attn_func`，不关心
  来源），所以拒绝的是**自动安装**，不是功能本身。
- **onnxruntime GPU EP**：DCU 上没有 `CUDAExecutionProvider`，海光提供 MIGraphX EP。
  那套「预加载 `site-packages/nvidia/*/lib/*.so`」在 DCU 上整段跳过。

**监控层**：NVML 是 NVIDIA 专有，DCU 上 init 不了。`free_vram_bytes()` /
`device_stats()` 按后端选路径：NVIDIA 优先 NVML，DCU 走 torch `mem_get_info`。

NVIDIA 上**必须**保留 NVML 优先：WDDM 下 `cudaMemGetInfo` 是每进程虚拟化视角，看不到
其他进程占用（真机实测：他进程持有 20 GB 时它仍报全量 free），用它做跨进程护栏形同虚设。
DCU 上 torch 就够——DCU 只在 Linux 跑，没有 WDDM 那套虚拟化，`mem_get_info` 本身即全卡
口径。

代价：DCU 上利用率与温度当前不报（要解析 hy-smi 文本，无稳定机器可读接口）。
`GpuStats.util_pct` 因此从 `int` 放宽为 `int | None`，前端在 null 时隐藏利用率 pill
（不能用 0% 兜底——0% 是合法读数），显存 pill 不受影响。

## 理由

- **方案 3 比方案 2 省的不是当下的行数，是后续的正确性。** 判定散开后，「DCU 上不能 pip
  装 torch」这类护栏只要漏一处就等于没有——而漏掉的代价是用户容器报废。收敛到一处后，
  漏掉的可能性从「十几个调用点都要记得」变成「一个函数返回 False」。
- **上层接口用能力查询而不是后端枚举**，是因为真正的变化点是能力矩阵而非厂商数量。
  `supports_xformers()` 这个名字让调用点自解释，也让「NVIDIA 上用户没装 xformers」与
  「DCU 上装不了 xformers」这两种情况自然走同一条降级路径。
- **训练主链路一行不改是这次移植成本低的根本原因**，值得写下来：如果项目此前用了自定义
  CUDA kernel、依赖 `_scaled_mm` 做 fp8、或到处硬编码 `torch.version.cuda`，代价会完全
  不同。fp8 走 dequant 而非原生 matmul 原本是为了对齐 ComfyUI 逐位一致，意外地成了可移植
  性红利。
- **不可用能力选择 fail-fast 而非静默降级**：NaViT 静默退到 dense mask 会让用户以为省了
  显存实际没省（与 §3.4 里 `blocks_to_swap` 的既有处理同源）；flash-attn 装了 CUDA wheel
  到 DCU 会让 `current_status()` 报「已安装」而训练时才崩，排错成本远高于当场拒绝。

## 影响

- 新增 `utils/accelerator.py`（唯一权威源）与 `tools/probe_accelerator.py`（环境探测，
  移植与排错时采集基线）。
- `requirements.txt` 的 `nvidia-ml-py` 保留：纯 Python wrapper，在所有平台装得上，DCU 上
  init 失败后走 torch 路径，不值得为它拆分 requirements 文件。
- README 硬件要求、`docs/user-guide/getting-started.md` 增补 DCU 段。
- 遗留：DCU 上的利用率 / 温度监控待 hy-smi 输出格式确认后补
  （`tools/probe_accelerator.py` 的 `smi` 段原样保留整段 stdout 正为此准备）。
