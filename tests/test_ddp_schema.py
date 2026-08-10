"""多卡（DDP）schema 字段 + block swap 互斥声明。

`ddp_num_processes` 的默认值 1 是整条多卡改动的安全阀：cmd_builder 只在 >1 时
套 torchrun，utils/distributed 在 world_size==1 时所有 is_main() 门控恒真。
默认值一旦被改动，所有单卡用户会静默进入多进程路径 —— 本文件第一条就锁它。

batch_size 的「每卡」语义也在此锁死：它决定全局 batch 的换算口径
（batch_size × grad_accum × ddp_num_processes），改口径等于悄悄换掉用户的
有效超参（开 2 卡后全局 batch 翻倍、lr 不再匹配）。
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from studio.domain.config_rules import apply_disable_rule_fixes, iter_pin_rules
from studio.schema import TrainingConfig


# ------------------------------------------------------------- 默认值与取值域

def test_default_is_single_process() -> None:
    """默认单卡 —— 现有行为不变的唯一保证。"""
    assert TrainingConfig().ddp_num_processes == 1


def test_rejects_zero_and_negative() -> None:
    for bad in (0, -1):
        with pytest.raises(ValidationError):
            TrainingConfig(ddp_num_processes=bad)


def test_no_upper_bound_pinned_in_schema() -> None:
    """不在 schema 里钉卡数上限：上限是机器事实（还受 CUDA_VISIBLE_DEVICES /
    HIP_VISIBLE_DEVICES 影响），schema 是跨机器共享的预设，钉死会让 8 卡机
    读不了 2 卡机存的预设。真正的越界由训练侧 distributed.init() 报。"""
    assert TrainingConfig(ddp_num_processes=8).ddp_num_processes == 8


# --------------------------------------------------------------- UI 元信息契约

def test_field_is_visible_in_training_group() -> None:
    """归 training 组（它改训练数值语义），且不能是 hidden —— hidden 字段
    SchemaForm 直接跳过，用户在 UI 上永远看不到这个开关。"""
    extra = TrainingConfig.model_fields["ddp_num_processes"].json_schema_extra
    assert extra["group"] == "training"
    assert extra.get("hidden") is not True
    assert extra["advanced"] is True  # 多卡是进阶配置，简单模式不铺噪音


def test_description_states_per_card_batch_semantics() -> None:
    """描述必须写明「每卡」+ 全局 batch 换算 —— 这是 UI tooltip 的唯一来源，
    漏了用户开 2 卡后不知道全局 batch 已翻倍（lr 随之失配）。"""
    desc = TrainingConfig.model_fields["ddp_num_processes"].description or ""
    assert "每卡" in desc
    assert "grad_accum" in desc and "batch_size" in desc
    # 可见卡数上限与两个后端的可见性变量都要提到（用户自查线索）
    assert "CUDA_VISIBLE_DEVICES" in desc and "HIP_VISIBLE_DEVICES" in desc


# ------------------------------------------------------ 与 block swap 的互斥

def test_block_swap_rule_is_declared() -> None:
    """互斥走 disable_when 声明（而非手写 validator），这样前端灰显 + 后端
    校验 + tolerant 修复三处从同一份声明派生。"""
    rules = {name: (expr, pin) for name, expr, pin, _h in iter_pin_rules(TrainingConfig)}
    assert rules["blocks_to_swap"][0] == "ddp_num_processes!=1"
    assert rules["blocks_to_swap"][1] == 0


def test_single_gpu_keeps_block_swap_usable() -> None:
    """回归：单卡下 block swap 一切照旧（互斥只在多卡时生效）。"""
    assert TrainingConfig(blocks_to_swap=28).blocks_to_swap == 28


def test_multi_gpu_pins_block_swap_when_absent() -> None:
    """只开多卡、没提 blocks_to_swap → 落钉值 0，不报错。"""
    assert TrainingConfig(ddp_num_processes=2).blocks_to_swap == 0


def test_multi_gpu_with_explicit_block_swap_is_rejected() -> None:
    """显式两个都开 → fail-fast，绝不静默改用户配置。"""
    with pytest.raises(ValidationError, match="blocks_to_swap"):
        TrainingConfig(ddp_num_processes=2, blocks_to_swap=28)


def test_tolerant_fix_drops_block_swap_not_ddp() -> None:
    """存量 config 读盘修复：钉 blocks_to_swap=0、保住多卡设置
    （ddp_num_processes 不在 TOLERANT_FIX_GATE_FIRST）。"""
    fixed, names = apply_disable_rule_fixes(
        {"ddp_num_processes": 2, "blocks_to_swap": 28}, TrainingConfig
    )
    assert fixed["blocks_to_swap"] == 0
    assert fixed["ddp_num_processes"] == 2
    assert "blocks_to_swap" in names


# ------------------------------------------- PPSF fused backward × 多卡互斥


def test_ppsf_fused_back_pass_rule_is_declared() -> None:
    """PPSF fused backward 必须声明成多卡 disable 规则，钉 False。

    为什么是 schema 规则而不是只靠 runtime 检查：声明在这里，UI 灰显、后端
    fail-fast、tolerant 读盘修复三处从同一份声明派生（与 blocks_to_swap 同款）。
    """
    rules = {name: (expr, pin) for name, expr, pin, _ in iter_pin_rules(TrainingConfig)}
    assert "ppsf_fused_back_pass" in rules, (
        "ppsf_fused_back_pass 没有 disable_when 声明 —— 多卡下开它会静默地让各 rank "
        "梯度不一致"
    )
    assert rules["ppsf_fused_back_pass"][0] == "ddp_num_processes!=1"
    assert rules["ppsf_fused_back_pass"][1] is False


def test_single_gpu_keeps_ppsf_fused_back_pass_usable() -> None:
    """单卡下它照常可用 —— 这是个真有用的省显存开关，不能一并禁掉。"""
    assert TrainingConfig(ppsf_fused_back_pass=True).ppsf_fused_back_pass is True


def test_multi_gpu_pins_ppsf_fused_back_pass_when_absent() -> None:
    """只开多卡、没提这个字段 → 落钉值 False，不报错。"""
    assert TrainingConfig(ddp_num_processes=2).ppsf_fused_back_pass is False


def test_multi_gpu_with_explicit_ppsf_fused_back_pass_is_rejected() -> None:
    """显式同时开两者 → 校验失败（fail-fast，而不是静默钉值）。

    静默改用户显式写的值同样危险 —— 用户会以为省显存生效了。
    """
    with pytest.raises(ValidationError, match="ppsf_fused_back_pass"):
        TrainingConfig(ddp_num_processes=2, ppsf_fused_back_pass=True)


def test_tolerant_fix_drops_ppsf_fused_back_pass_not_ddp() -> None:
    """存量 config 读盘修复：钉 ppsf_fused_back_pass=False、保住多卡设置。

    方向很重要 —— 用户开多卡是为了摊算力，不能反过来把多卡关掉。
    """
    fixed, _ = apply_disable_rule_fixes(
        {"ddp_num_processes": 2, "ppsf_fused_back_pass": True}, TrainingConfig
    )
    assert fixed["ppsf_fused_back_pass"] is False
    assert fixed["ddp_num_processes"] == 2
