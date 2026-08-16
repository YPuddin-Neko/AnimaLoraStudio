# DCU 多卡分支的 DDP 关键改动

v0.24.1 合并验证通过后，整理一份索引备查。以下全部改动**已在真机验证**（v0.23.1 多
卡本），本次合并零冲突、零丢失。

## 1. 后端：跨 rank 状态一致 + 集合操作同步

### 1.1 NaN loss 的全体一致判定（`training/loop.py`）

```python
def _agree_on_finite_loss(loss) -> bool:
    """多卡下取所有 rank 的一致结论（all_reduce 取与）。"""
```

- **位置**：`runtime/training/loop.py:191`
- **调用点**：`loop.py:641`（判断是否跳过当前 micro-batch）
- **用途**：某个 rank 遇到 NaN 时，所有 rank 一起跳过该 batch，避免等死在后续的 all_reduce

### 1.2 DDP adapter shim（`training/phases/models.py`）

```python
class _DDPAdapterSync:
    """DDP forward 包装：进入前 adapter_sync()，出去后还原。"""
```

- **位置**：`runtime/training/phases/models.py:374`
- **用途**：LyCORIS adapter 的多卡同步（进 DDP forward 前调 `adapter_sync()`，否则各 rank 的 dropout mask 不一致）
- **调用点**：`models.py:391`（DDP 包装处）

### 1.3 ppsf_fused_back_pass 与 DDP 冲突拦截（`training/phases/bootstrap.py`）

- **位置**：`runtime/training/phases/bootstrap.py:177`
- **检查逻辑**：启动期若检测到 `ppsf_fused_back_pass=true` + 多进程，直接报错退出
- **原因**：fused backward 会在反向过程中就地更新参数并释放梯度，与 DDP reducer 抢同一块内存，导致各 rank 梯度不一致（静默分歧）

### 1.4 异步暂停信号（`training/context.py`）

```python
pause_signal_seen: bool = False
```

- **位置**：`runtime/training/context.py:129`
- **用途**：单卡下 Ctrl+C 立即退，多卡下只置标志、等下一个 epoch 末退出（避免某个 rank 提前退出导致其他 rank 等死在集合操作）
- **调用点**：`context.py:296`（`request_pause_from_signal`）+ `loop.py` 两处轮询点

---

## 2. 后端：海光 DCU 硬件适配（`utils/accelerator.py`）

### 2.1 sysfs 功率采集（DCU 专用）

```python
def _sysfs_card_metrics(card_idx: int) -> dict[str, Any] | None:
    """从 /sys/class/drm/card{N}/device/* 读显存 / 功率 / 频率。"""
```

- **位置**：`utils/accelerator.py:158`
- **用途**：DCU 上 hy-smi 容器内的功率读数偏低 6-7 倍，改读 sysfs 的实时值
- **字段**：`power_w` / `power_cap_w` / `sclk_mhz` / `sclk_max_mhz`（见 `DeviceStats`）

### 2.2 DRM 卡枚举（`visible_drm_cards`）

```python
def visible_drm_cards() -> list[int]:
    """按 HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES 过滤可见的 /sys/class/drm/card* 索引。"""
```

- **位置**：`utils/accelerator.py:104`
- **用途**：多卡环境下只枚举当前进程可见的卡（避免读到别的容器的卡）

### 2.3 NVML active 判定修正（`_nvml_active_index`）

```python
def _nvml_active_index() -> int | None:
    """NVML 判断当前 rank 正在用哪张卡 —— 按进程 PID 的显存占用。"""
```

- **位置**：`utils/accelerator.py:256`
- **修正内容**：原逻辑只看 `torch.cuda.current_device()`，多卡 DDP 下每个 rank 的返回值都是 0（因为环境变量让各 rank 只看见一张卡）

---

## 3. 前端：功率 pill + 步数估算

### 3.1 功率 pill（`SystemStats.tsx`）

- **新增字段**：`power_w` / `power_cap_w` / `sclk_mhz` / `sclk_max_mhz`（`studio/services/system_stats.py` 传上来）
- **UI**：topbar 新增 POWER pill，显示合计功率 / 上限，hover 显示逐卡功率 + 频率
- **位置**：`studio/web/src/components/SystemStats.tsx:146-169`

### 3.2 步数估算逻辑修正（`trainSteps.ts`）

```typescript
export function estimateSteps(config: TrainingConfig): {
  totalSteps: number
  stepsPerEpoch: number
  reason?: string
} | null
```

- **位置**：`studio/web/src/lib/trainSteps.ts:10`
- **修正内容**：
  1. 单卡 / 多卡的 `effective_batch = batch_size * grad_accum * num_processes`
  2. `steps_per_epoch = ceil(num_images / effective_batch)`（向上取整，确保最后一组不足一个 batch 时也算一步）
  3. `total_steps = steps_per_epoch * num_epochs`
- **调用点**：训练配置页的步数预览、queue 任务详情页

---

## 4. 测试覆盖

全部通过（本地 Windows 无 torch 环境，只跑了不依赖 torch 的套件）：

```bash
pytest tests/test_accelerator.py           # accelerator 三后端 mock
pytest tests/test_system_stats.py          # system_stats 三态映射 + 采样线程
pytest tests/test_sysmem.py                # sysmem 峰值追踪
pytest tests/test_ddp_prerequisites.py     # DDP 先决条件检查（bootstrap）
pytest tests/test_collective_gating.py     # 集合操作门控（_agree_on_finite_loss）
pytest tests/test_pause_marker.py          # 暂停标记文件逻辑
```

**待真机补验**（需 torch）：
- `tests/test_block_swap_pinned_release.py`（v0.24.1 新增，测 XY 换 LoRA 的 pinned tensor 释放）
- `tests/test_block_swap_fp8_merge.py`（v0.24.1 补了一条，测 fp8 LoRA merge 的显存占用）

---

## 5. 文档更新

- **README.md / README.en.md**：「海光 DCU」章节 4 处 / 「Hygon DCU」3 处，描述 DTK 环境 / flash-attn / xformers / NaViT 打包 / onnxruntime GPU EP 的注意事项
- **ADR 0016**：`docs/adr/0016-dual-accelerator-backend-hygon-dcu.md`，accelerator 抽象层 + DCU 后端设计

---

## 关键命令速查

```bash
# 后端自查
python tools/probe_accelerator.py    # 后端识别、设备信息、fp8 / SDPA / block swap 实测
bash tools/find_flash_attn.sh        # 找本机有没有现成的 flash-attn 包 / .so

# 多卡启动
torchrun --nproc_per_node=2 runtime/train.py @config.txt

# 测试
pytest tests/test_ddp_prerequisites.py tests/test_collective_gating.py -v
```

---

**本文档最后更新**：v0.24.1 合并后（3f7e8f7）
