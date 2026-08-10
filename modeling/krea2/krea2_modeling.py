"""Krea 2 single-stream MMDiT with ComfyUI-compatible module names.

Core structure derived from ComfyUI ``comfy/ldm/krea2/model.py`` (GPL-3.0):
Copyright Comfy-Org and ComfyUI contributors.
https://github.com/comfyanonymous/ComfyUI/blob/87d23b81765161624889febfb3b81f19f3c8435b/comfy/ldm/krea2/model.py

Training-oriented tensor layout, RoPE, and checkpointing were adapted from
kohya-ss/musubi-tuner ``krea2_mmdit.py`` (Apache-2.0):
Copyright 2026 Kohya S. and musubi-tuner contributors.
https://github.com/kohya-ss/musubi-tuner/blob/8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6/src/musubi_tuner/krea2/krea2_mmdit.py

Pinned source revisions and file URLs are recorded in ``THIRD_PARTY_NOTICES.md``.
This file is distributed under the repository's GPL-3.0 license.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


#: 带 mask 的注意力按 query 维度分块时，每块的 query 数。
#:
#: 只在**不得不 materialize 分数矩阵**时才起作用（带 attn_mask → flash 不接、
#: mem-efficient 在 DTK 上没编译 → 退到 math 后端）。math 后端的峰值是
#: ``[B, H, S_q, S_k]``，把 S_q 切成块后峰值降到 ``[B, H, chunk, S_k]``。
#:
#: 兜底块大小。拿不到显存读数时用它（保守值，见 :func:`_pick_query_chunk`）。
_MASKED_ATTN_QUERY_CHUNK = 256

#: 峰值 / 单个分数矩阵：预算按 ``理论大小 × 该系数`` 算。
#:
#: **单次分配**已由真机两个观测点精确定出为 1.0 倍（BW1000 / DTK 26.04 / torch
#: 2.5.1，bucket 2624×1472 → S_k = 15088 + 512 = 15600，H=48，B=2）::
#:
#:     chunk=1024 → B×H×chunk×S_k×4B = 5.713 GiB，torch 报 5.71 GiB
#:     chunk= 512 → B×H×chunk×S_k×4B = 2.856 GiB，torch 报 2.86 GiB
#:
#: 两点都对得上小数点后两位，所以「一次分配 = 恰好一份 fp32 分数矩阵」是确证的，
#: 不是拟合。（早先写的 1.84 倍是错的：那次把 S_k 当成了另一个 bucket 的 16928、
#: 又把 B 当成 1，两个错误凑出一个假倍数。）
#:
#: 但**峰值**不等于单次分配：softmax 会另开一份输出张量，两份同量级张量短时间共存，
#: 而 OOM 只报那次失败的分配。预算要按峰值算，所以取 2.0。
#:
#: 选小了只是多几次 kernel 启动，选大了直接 OOM —— 这个方向上宁可保守。
_MATH_SDPA_OVERHEAD = 2.0

#: 分数矩阵最多吃掉当前空闲显存的比例。
#:
#: 不敢用满是因为 attention 内部除了分数矩阵还有输出张量与临时量，而 allocator 的
#: 碎片也让「空闲」不等于「可连续分配」。0.5 在真机上留出了足够的安全边际。
_CHUNK_FREE_VRAM_FRACTION = 0.5


def _pick_query_chunk(q: Tensor, s_k: int) -> int:
    """按**当前空闲显存**选 query 块大小；拿不到读数时回落
    :data:`_MASKED_ATTN_QUERY_CHUNK`。

    为什么不写死一个常数：可用余量在不同配置下差一个数量级，写死必然二选一地错。
    真机对照（rank 1，2048px 桶 S≈16.9k，H=48）——

    - LoKr **full matrix**（用户刻意选的训练方式）：803M 可训练参数，加 PPSF 的两份
      state（各 2.99 GiB）与 DiT 权重 24.12 GiB → 常驻 33 GiB，激活实测再吃 22 GiB，
      OOM 那一刻只剩 2.8 GiB。这里只能取到下限 128。
    - 常规低秩 LoKr（rank 32 级）：可训练参数几十 M，常驻不到 26 GiB，余量 20 GiB 以上
      → chunk 选到 768，kernel 启动次数只有下限那条路的 1/6。

    所以按运行时余量算：``chunk = free × fraction / (B × H × S_k × 4B × overhead)``。

    ``mem_get_info`` 是微秒级、attention 是毫秒级，每次调用问一次不值得优化掉 ——
    而缓存反而危险：同一进程里余量会随 block swap / 采样 / 其他 rank 的活动变化，
    用陈旧读数选块正是要避免的事。

    返回值向下取到 128 的倍数并夹在 ``[128, 4096]``：太小则 kernel 启动开销占比失控，
    太大则超出 math 后端本身能处理的合理范围（也没必要 —— 余量再多也不该一次吃 4 GiB
    以上，那说明该走 flash 而不是 math）。
    """
    fallback = _MASKED_ATTN_QUERY_CHUNK
    if not q.is_cuda:
        return fallback
    try:
        free, _total = torch.cuda.mem_get_info(q.device)
    except Exception:  # noqa: BLE001  拿不到读数就用保守兜底，不能让选块本身崩
        return fallback
    # 分数矩阵是 [B, H, chunk, S_k]，所以 batch 与 head 都得算进单位成本 —— 漏掉 batch
    # 会在 bs=2 时把预算高估一倍，正好落在 OOM 那一侧。
    batch = q.shape[0] if q.ndim >= 4 else 1
    per_query = batch * q.shape[-3] * s_k * 4 * _MATH_SDPA_OVERHEAD
    if per_query <= 0:
        return fallback
    budget = free * _CHUNK_FREE_VRAM_FRACTION
    chunk = int(budget // per_query)
    chunk = (chunk // 128) * 128          # 向下取到 128 的倍数
    return max(128, min(chunk, 4096))


def _chunked_masked_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    chunk: int | None = None,
) -> Tensor:
    """按 query 维度分块的 SDPA，**数值上与不分块完全等价**。

    为什么可以这么做：softmax 沿 **key** 维归一化，每个 query 行的输出只依赖该行
    自己的分数向量 —— query 之间没有任何耦合。所以沿 query 切块是**精确**的，不是
    近似、也不需要像 flash 那样做 online-softmax 的 rescale。

    为什么需要这么做：带 ``attn_mask`` 时 SDPA 只能走 math 后端（flash 不支持任意
    mask，mem-efficient 在海光 DTK 上编译时没开），而 math 会显式 materialize
    ``[B, H, S_q, S_k]``。Krea2 在 2048px 桶上 S≈16.9k、48 heads，bs=2 就是
    102 GiB —— 64GB 卡必 OOM，且这与 batch 无关：bs=1 也要 51 GiB，加上约 30GB
    常驻权重同样装不下。分块把峰值压到 ``chunk/S_q`` 倍。

    ``mask`` 是 ``[B, 1, 1, S_k]`` 的 key-padding mask（bool，True = 参与）。它在
    query 维上是广播的，所以每块直接复用同一个 mask，无需切片 —— 这也是为什么这里
    只切 query 不切 key。

    ``q`` 的 query 数不超过 ``chunk`` 时直接走原路，不付分块的额外开销。

    ``chunk=None``（默认）时按当前空闲显存自适应选块，见 :func:`_pick_query_chunk`；
    传显式值只给测试用（要在小张量上强制走分块路径）。
    """
    s_q = q.shape[-2]
    if chunk is None:
        chunk = _pick_query_chunk(q, k.shape[-2])
    if s_q <= chunk:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False,
        )
    outs = []
    for start in range(0, s_q, chunk):
        outs.append(
            F.scaled_dot_product_attention(
                q[..., start:start + chunk, :],
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
            )
        )
    return torch.cat(outs, dim=-2)


def _rope(pos: Tensor, dim: int, theta: float) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    angles = torch.einsum("...n,d->...nd", pos, omega)
    matrix = torch.stack(
        [torch.cos(angles), -torch.sin(angles), torch.sin(angles), torch.cos(angles)],
        dim=-1,
    )
    return rearrange(matrix, "b n d (i j) -> b n d i j", i=2, j=2).float()


def _apply_rope(q: Tensor, k: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    q_float = q.float().reshape(*q.shape[:-1], -1, 1, 2)
    k_float = k.float().reshape(*k.shape[:-1], -1, 1, 2)
    matrix = freqs[:, None, :, :, :]
    q_out = matrix[..., 0] * q_float[..., 0] + matrix[..., 1] * q_float[..., 1]
    k_out = matrix[..., 0] * k_float[..., 0] + matrix[..., 1] * k_float[..., 1]
    return q_out.reshape_as(q).to(q.dtype), k_out.reshape_as(k).to(k.dtype)


def _timestep_embedding(t: Tensor, dim: int, *, dtype: torch.dtype) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(1e4)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    angles = (t.float() * 1e3)[:, None, None] * freqs
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1).to(dtype)


@dataclass(frozen=True)
class Krea2Config:
    """Architecture values for the public Krea-2 Raw/Turbo checkpoints."""

    features: int = 6144
    tdim: int = 256
    txtdim: int = 2560
    heads: int = 48
    multiplier: int = 4
    layers: int = 28
    patch: int = 2
    channels: int = 16
    bias: bool = False
    theta: float = 1e3
    kvheads: int = 12
    txtlayers: int = 12
    txtheads: int = 20
    txtkvheads: int = 20

    def __post_init__(self) -> None:
        if self.features <= 0 or self.features % self.heads:
            raise ValueError("Krea2 features 必须为正数且能被 heads 整除")
        if self.heads % self.kvheads:
            raise ValueError("Krea2 heads 必须能被 kvheads 整除")
        if self.txtdim <= 0 or self.txtdim % self.txtheads:
            raise ValueError("Krea2 txtdim 必须为正数且能被 txtheads 整除")
        if self.txtheads % self.txtkvheads:
            raise ValueError("Krea2 txtheads 必须能被 txtkvheads 整除")
        if self.tdim <= 0 or self.tdim % 2:
            raise ValueError("Krea2 tdim 必须为正偶数")
        if min(self.layers, self.patch, self.channels, self.txtlayers) <= 0:
            raise ValueError("Krea2 layers/patch/channels/txtlayers 必须为正数")
        if self.theta <= 0:
            raise ValueError("Krea2 theta 必须为正数")


KREA2_CONFIG = Krea2Config()


class RMSNorm(nn.Module):
    """RMSNorm using Krea2's zero-centered ``1 + scale`` convention."""

    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(features, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        weight = self.scale.float() + 1.0
        return F.rms_norm(
            x.float(),
            (self.features,),
            weight=weight,
            eps=self.eps,
        ).to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        return self.qnorm(q), self.knorm(k)


class SwiGLU(nn.Module):
    def __init__(
        self,
        features: int,
        multiplier: int,
        bias: bool = False,
        multiple: int = 128,
    ):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        kvheads: int | None = None,
        bias: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.kvheads = heads if kvheads is None else kvheads
        if dim % heads or heads % self.kvheads:
            raise ValueError("Krea2 attention 维度或 GQA heads 不可整除")
        self.headdim = dim // heads
        self.wq = nn.Linear(dim, self.headdim * heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(
        self,
        x: Tensor,
        freqs: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        q = rearrange(self.wq(x), "b l (h d) -> b h l d", h=self.heads)
        k = rearrange(self.wk(x), "b l (h d) -> b h l d", h=self.kvheads)
        v = rearrange(self.wv(x), "b l (h d) -> b h l d", h=self.kvheads)
        gate = self.gate(x)

        q, k = self.qknorm(q, k)
        if freqs is not None:
            q, k = _apply_rope(q, k, freqs)
        if self.kvheads != self.heads:
            repeat = self.heads // self.kvheads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        if mask is None:
            # 无 mask 是快路径：flash 后端接管，分数矩阵根本不 materialize。
            # 上游的 forward 已尽量把「全 True 的 padding mask」折成 None，让这条
            # 路径尽可能常被命中。
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
            )
        else:
            # 有真实 padding 时只能走 math 后端 → 按 query 分块限制峰值。
            # 数值与不分块逐值等价（softmax 沿 key 归一化，query 行之间无耦合）。
            out = _chunked_masked_attention(q, k, v, mask)
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.wo(out * torch.sigmoid(gate))


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim))

    def forward(self, vec: Tensor) -> tuple[Tensor, Tensor]:
        return (vec + self.lin.unsqueeze(0)).chunk(2, dim=1)


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, vec: Tensor) -> tuple[Tensor, ...]:
        return (vec + self.lin).chunk(6, dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, axes_dim: tuple[int, int, int], theta: float):
        super().__init__()
        self.axes_dim = axes_dim
        self.theta = theta

    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat(
            [_rope(pos[..., axis], dim, self.theta) for axis, dim in enumerate(self.axes_dim)],
            dim=-3,
        )


class TextFusionBlock(nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int | None = None,
    ):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.prenorm(x), mask=mask)
        return x + self.mlp(self.postnorm(x))


class TextFusionTransformer(nn.Module):
    def __init__(
        self,
        num_txt_layers: int,
        txt_dim: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int | None = None,
    ):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, seq_len, num_layers, dim = x.shape
        x = x.reshape(batch * seq_len, num_layers, dim)
        for block in self.layerwise_blocks:
            x = block(x.contiguous())
        x = rearrange(x, "(b l) n d -> b l d n", b=batch, l=seq_len)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int | None = None,
    ):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        freqs: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        attn_in = (1 + prescale) * self.prenorm(x) + preshift
        x = x + pregate * self.attn(attn_in, freqs=freqs, mask=mask)
        mlp_in = (1 + postscale) * self.postnorm(x) + postshift
        return x + postgate * self.mlp(mlp_in)


class LastLayer(nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        return self.linear((1 + scale) * self.norm(x) + shift)


class SingleStreamDiT(nn.Module):
    """Krea2 DiT; parameter paths intentionally match ComfyUI exactly."""

    def __init__(self, config: Krea2Config = KREA2_CONFIG):
        super().__init__()
        self.config = config
        head_dim = config.features // config.heads
        axes = (
            head_dim - 12 * (head_dim // 16),
            6 * (head_dim // 16),
            6 * (head_dim // 16),
        )
        if sum(axes) != head_dim or any(dim <= 0 or dim % 2 for dim in axes):
            raise ValueError(f"Krea2 RoPE axes 非法：axes={axes}, head_dim={head_dim}")

        self.posemb = PositionalEncoding(axes, theta=config.theta)
        self.first = nn.Linear(
            config.channels * config.patch**2,
            config.features,
            bias=True,
        )
        self.blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    config.features,
                    config.heads,
                    config.multiplier,
                    config.bias,
                    config.kvheads,
                )
                for _ in range(config.layers)
            ]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(config.tdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.txtfusion = TextFusionTransformer(
            config.txtlayers,
            config.txtdim,
            config.txtheads,
            config.multiplier,
            config.bias,
            config.txtkvheads,
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(config.txtdim),
            nn.Linear(config.txtdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.last = LastLayer(config.features, config.patch, config.channels)
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features * 6),
        )
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def _normalize_context(self, context: Tensor) -> Tensor:
        config = self.config
        if context.ndim == 3:
            expected = config.txtlayers * config.txtdim
            if context.shape[-1] != expected:
                raise ValueError(
                    f"Krea2 context 最后一维应为 {expected}，实际 {context.shape[-1]}"
                )
            return context.reshape(
                context.shape[0],
                context.shape[1],
                config.txtlayers,
                config.txtdim,
            )
        if context.ndim != 4 or context.shape[-2:] != (
            config.txtlayers,
            config.txtdim,
        ):
            raise ValueError(
                "Krea2 context 应为 (B,L,txtlayers,txtdim) 或对应的扁平三维 tensor"
            )
        return context

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        context: Tensor,
        attention_mask: Tensor | None = None,
        *,
        use_checkpoint: bool = False,
    ) -> Tensor:
        temporal = x.ndim == 5
        if temporal:
            if x.shape[2] != 1:
                raise ValueError("Krea2 v1 只支持 T==1 的 5D latent")
            x = x.squeeze(2)
        if x.ndim != 4:
            raise ValueError("Krea2 latent 应为 (B,C,H,W) 或 (B,C,1,H,W)")
        if x.shape[1] != self.config.channels:
            raise ValueError(
                f"Krea2 latent channels 应为 {self.config.channels}，实际 {x.shape[1]}"
            )

        context = self._normalize_context(context)
        batch, _, original_h, original_w = x.shape
        if context.shape[0] != batch:
            raise ValueError("Krea2 latent 与 context batch size 不一致")
        if timesteps.ndim != 1 or timesteps.numel() != batch:
            raise ValueError("Krea2 timesteps 应为 batch 长度的一维 tensor")

        patch = self.config.patch
        pad_h = (-original_h) % patch
        pad_w = (-original_w) % patch
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        height, width = x.shape[-2] // patch, x.shape[-1] // patch

        image = rearrange(
            x,
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=patch,
            pw=patch,
        )
        image = self.first(image)
        time = self.tmlp(
            _timestep_embedding(
                timesteps,
                self.config.tdim,
                dtype=image.dtype,
            )
        )
        time_mod = self.tproj(time)

        text_len = context.shape[1]
        text_mask = None
        combined_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch, text_len):
                raise ValueError("Krea2 attention_mask 应为 (B,text_len)")
            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)
            # 全 True 的 key-padding mask 与 None 语义完全相同（SDPA 的 bool mask
            # 约定：True = 参与注意力），但**代价差一个数量级**：带 attn_mask 时
            # flash 后端压根不接（它不支持任意 mask）、mem-efficient 在部分平台也不
            # 可用，SDPA 于是退到 math 后端，把 [B, 1, 1, S] 广播成完整的
            # [B, H, S, S] 分数矩阵显式 materialize。
            #
            # 真机实测（BW1000 64GB / Krea2 / bs=1，DTK 无 mem-efficient 后端）：
            # 这一条退化让单次 attention 试图分配 42.99 GiB 直接 OOM；置 None 走
            # flash 后同一个前向只需常数级额外显存。
            #
            # 什么时候会全 True：`pad_text_conditions` 按 batch 内**最长** caption
            # 右填充，所以 bs=1 时恒全 True（只有一条 caption，max_length 就是它自己
            # 的长度）；bs>1 时各 caption 长度恰好一致同样全 True。Krea2 是 12.9B
            # 模型、bs=1 是常态，所以这条捷径命中率很高。
            #
            # 只需判文本段：图像 token 无 padding（image_mask 恒 ones），所以
            # combined 是否全 True 完全由 attention_mask 决定。一次 GPU reduce +
            # D2H 同步，相对省下的 materialize 可以忽略。
            if bool(attention_mask.all()):
                # 短路：text_mask / combined_mask 都保持 None，连 image_mask 的
                # 分配与 torch.cat 都不做 —— 那两步在这条（最常见的）路径上纯浪费。
                pass
            else:
                text_mask = attention_mask[:, None, None, :]
                image_mask = torch.ones(
                    batch,
                    image.shape[1],
                    device=x.device,
                    dtype=torch.bool,
                )
                combined_mask = torch.cat((attention_mask, image_mask), dim=1)
                combined_mask = combined_mask[:, None, None, :]

        text = self.txtfusion(context, mask=text_mask)
        text = self.txtmlp(text)
        image_len = image.shape[1]
        combined = torch.cat((text, image), dim=1)

        text_pos = torch.zeros(
            batch,
            text_len,
            3,
            device=x.device,
            dtype=torch.float32,
        )
        image_pos = torch.zeros(
            height,
            width,
            3,
            device=x.device,
            dtype=torch.float32,
        )
        image_pos[..., 1] = torch.arange(height, device=x.device)[:, None]
        image_pos[..., 2] = torch.arange(width, device=x.device)[None, :]
        image_pos = image_pos.reshape(1, image_len, 3).expand(batch, -1, -1)
        freqs = self.posemb(torch.cat((text_pos, image_pos), dim=1))

        checkpoint_blocks = (
            (use_checkpoint or self.gradient_checkpointing)
            and self.training
            and torch.is_grad_enabled()
        )
        for block in self.blocks:
            if checkpoint_blocks:
                from torch.utils.checkpoint import checkpoint

                def custom_forward(hidden: Tensor, current=block) -> Tensor:
                    return current(hidden, time_mod, freqs, combined_mask)

                combined = checkpoint(custom_forward, combined, use_reentrant=False)
            else:
                combined = block(combined, time_mod, freqs, combined_mask)

        final = self.last(combined, time)
        output = final[:, text_len:text_len + image_len]
        output = rearrange(
            output,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=height,
            w=width,
            ph=patch,
            pw=patch,
            c=self.config.channels,
        )
        output = output[:, :, :original_h, :original_w]
        return output.unsqueeze(2) if temporal else output
