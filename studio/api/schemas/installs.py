"""安装 / runtime 类 endpoint 请求 BaseModel（PR-6 commit 3 从 server.py 抽出）。

涵盖 wd14 / torch / flash-attention / llm-tagger 域。xformers 无请求 body。

响应体这边**没有** BaseModel：这些端点历史上一路返回 service 层的原始
``dict[str, Any]``（key 由 service 决定，前端 client.ts 手写对应 interface）。
本文件只补一个 :func:`accelerator_block` —— 四个域的响应里都要带同一份后端信息，
让前端能按后端置灰不适用的按钮，而不是让用户点了才收 500。放在 schemas 层是因为
它描述的是 **API 契约的形状**，不是任何 service 的内部状态。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel

from utils import accelerator

logger = logging.getLogger(__name__)


class WD14InstallRequest(BaseModel):
    target: str = "auto"  # "auto" | "gpu" | "cpu" | "directml"


class TorchReinstallRequest(BaseModel):
    target: str = "auto"  # "auto" | "cu128" | "cu126" | "cu124" | "cu118" | "cpu"


class FlashAttnInstallRequest(BaseModel):
    url: Optional[str] = None  # None = 自动从 GitHub Releases 选最优


class LLMModelsRefreshRequest(BaseModel):
    # preset_id 指定要更新的 preset；不传则用当前 current_preset
    preset_id: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: Optional[int] = None


class LLMConnectionTestRequest(BaseModel):
    preset_id: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    timeout: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


# ---------------------------------------------------------------------------
# 后端能力块（四个装包域共用）
# ---------------------------------------------------------------------------


#: 探测彻底失败时的兜底：报 CUDA + 全部能力可用。**刻意不报 cpu / 全 False**——
#: 那会让所有安装按钮在一次偶发探测异常后集体置灰，用户完全没有自救入口。
#: 报 cuda 等于「维持移植前的行为」，最坏情况只是 DCU 上按钮没灰、点了拿到
#: service 层的明确 RuntimeError（service 侧有独立的同一道判断，不依赖本块）。
_ACCEL_FALLBACK: dict[str, Any] = {
    "backend": "cuda",
    "vendor_label": accelerator.VENDOR_LABEL["cuda"],
    "available": False,
    "device_count": 0,
    "device_names": [],
    "torch_version": None,
    "cuda_version": None,
    "hip_version": None,
    "gcn_arch": [],
    "import_error": None,
    "capabilities": {
        "manage_torch_install": True,
        "prebuilt_flash_attn_wheels": True,
        "xformers": True,
        "onnx_gpu_provider": "CUDAExecutionProvider",
    },
}


def accelerator_block() -> dict[str, Any]:
    """当前后端事实 + 各装包能力开关，挂到装包类 GET / POST 响应里。

    结构 = ``AcceleratorInfo.as_dict()`` 全字段 + ``capabilities`` 子对象：

    - ``capabilities.manage_torch_install``：本项目是否该替用户装 / 重装 torch。
      DCU 上 False（DTK torch 由镜像预装，pip 覆盖会报废环境）→ 前端置灰
      「重装 PyTorch」。
    - ``capabilities.prebuilt_flash_attn_wheels``：GitHub prebuilt wheel 那条路
      是否可用。DCU 上 False → 前端置灰 flash_attn 安装按钮 + 显示 DTK 渠道说明。
    - ``capabilities.xformers``：DCU 上 False → 置灰 xformers 安装按钮。
    - ``capabilities.onnx_gpu_provider``：当前后端的 GPU EP 名（``null`` = 没有
      GPU EP，打标只能跑 CPU）。

    这些布尔值**不是前端自己按 backend 推**的：单一权威源在
    ``utils/accelerator.py``，前端只消费结论。前端置灰是**体验优化**，真正的拦截
    在各 service 的 install() 里（同样问 accelerator），两层判断来自同一个源。
    """
    try:
        info = accelerator.detect()
        return {
            **info.as_dict(),
            "capabilities": {
                "manage_torch_install": accelerator.should_manage_torch_install(),
                "prebuilt_flash_attn_wheels": (
                    accelerator.supports_prebuilt_flash_attn_wheels()
                ),
                # 语义是「能不能 pip 自动装」，不是「xformers 能不能用」 —— DCU 上
                # 二者不同：公开源没 DCU wheel（自动装不了），但海光在光合社区发布
                # 配套 wheel，手动装上就可用。前端据此把「一键安装」置灰并给手动安装
                # 引导，**不要**据此说「不支持 xformers」。
                "xformers": accelerator.can_pip_install_xformers(),
                "onnx_gpu_provider": accelerator.onnx_gpu_provider(),
            },
        }
    except Exception:  # noqa: BLE001
        # 这是 Settings 页 mount 就拉的诊断数据，宁可给兜底也不要 500 把整段 UI
        # 打成「加载失败」。真原因靠 server log 里的 traceback 排查。
        logger.exception("加速器信息探测失败，返回兜底块")
        return dict(_ACCEL_FALLBACK)


def merge_accelerator_block(payload: dict[str, Any]) -> dict[str, Any]:
    """给已自带 `accelerator` 的响应补齐 `capabilities` 子对象。

    存在的理由：``/api/torch/status`` 的 service 层（``runtime/torch.py``）自己就返回
    ``accelerator = AcceleratorInfo.as_dict()``，但那里没有 ``capabilities``。若在
    router 里简单 setdefault，这个端点的 `accelerator` 形状就与另外三个不同，前端得
    按端点分叉写类型。这里做**幂等合并**：service 的字段全部保留（它是权威），只在缺
    ``capabilities`` 时补上。

    payload 原地改并返回同一个 dict —— 调用点都是 `return merge_...(svc())`，没有别的
    引用会看到中间状态。
    """
    block = accelerator_block()
    existing = payload.get("accelerator")
    if isinstance(existing, dict):
        # service 的字段优先（同名不覆盖），只补 capabilities 这类它没有的
        payload["accelerator"] = {**block, **existing}
        payload["accelerator"].setdefault("capabilities", block["capabilities"])
    else:
        payload["accelerator"] = block
    return payload
