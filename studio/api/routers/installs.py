"""runtime 装包 / LLM tagger admin endpoints（PR-6 commit 3 从 server.py 抽出）。

10 routes，按底层组件分 5 个子域共一个 router：

  wd14 (onnxruntime)：     GET /api/wd14/runtime           当前装包 + 可用 EP
                           POST /api/wd14/install          切 GPU/CPU 包（同步 pip，要重启）

  torch：                  GET /api/torch/status           torch 当前状态 + 推荐 cu tag
                           POST /api/torch/reinstall       注册重装请求（启动期执行）

  flash-attention：        GET /api/flash-attention/status status + 候选 wheel 列表
                           POST /api/flash-attention/install 装 wheel（要重启）

  xformers：               GET /api/xformers/status
                           POST /api/xformers/install      pip 直装（要重启）

  llm-tagger admin：       POST /api/llm-tagger/models/refresh  拉 /models 写 preset.model_ids
                           POST /api/llm-tagger/test            连通性测试（不写 secrets）

合一 router 而非 5 router：路由数少（每域 2）+ 共用 install 模式 / 共享
restart_required 语义，单独 router 太碎。前端 Settings 页也是装在一个抽屉里。

**后端能力透出**：前四个域的 GET / POST 响应里都带一个 `accelerator` 块
（`schemas.installs.accelerator_block()`），内含 backend / vendor_label / 各安装
能力开关。**刻意不新增端点**：一来 `tests/_snapshots/studio_routes.json` 是路由
快照，加端点必然失配；二来前端本来就在这四个 section 各拉一次 status，把后端信息
搭车带回去比多一次请求好。前端据此置灰不适用的按钮 —— 但那只是体验优化，真正的
拦截在各 service 的 install() 里（同样问 utils.accelerator）。
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from ..schemas.installs import (
    FlashAttnInstallRequest,
    LLMConnectionTestRequest,
    LLMModelsRefreshRequest,
    TorchReinstallRequest,
    WD14InstallRequest,
    accelerator_block,
    merge_accelerator_block,
)
from ... import secrets
from ...domain.errors import DomainError, ValidationError
from ...services.runtime import (
    flash_attention as flash_attention_setup,
    onnxruntime as onnxruntime_setup,
    pending_install,
    torch as torch_setup,
    xformers as xformers_setup,
)

router = APIRouter()


# WD14 runtime / GPU 装包 (PP8) ---------------------------------------------


@router.get("/api/wd14/runtime")
def wd14_runtime() -> dict[str, Any]:
    """返回 onnxruntime 当前装的是哪个包 + 可用 EP + GPU 硬件检测 + 后端能力。

    `cuda_detect` 在 DCU 上走 hy-smi / rocm-smi（service 内部按后端分派），key 不变。
    """
    rt = onnxruntime_setup.current_runtime()
    return {
        **rt,
        "cuda_detect": onnxruntime_setup.detect_cuda(),
        "accelerator": accelerator_block(),
    }


@router.post("/api/wd14/install")
def wd14_install(body: WD14InstallRequest) -> dict[str, Any]:
    """切换 onnxruntime 包：先 uninstall 两个互斥包，再装目标。

    同步 pip install，几分钟级；前端按钮要带 loading。
    onnxruntime 是 C extension，装完后**必须重启 Studio** 才能切换 EP（pip 卸装
    重装不能热替换已 import 的 .pyd/.so）。返回 `restart_required=True` 让前端
    显式提示。

    DCU 上 service 层会拒绝 `gpu` / `directml`（CUDA / DX12 build），以及「已装 DTK
    配套 onnxruntime 时的任何装包」（pip 会把它卸掉且装不回来）—— 都走下面那条
    RuntimeError → 500，message 里带完整原因与下一步。
    """
    if body.target not in ("auto", "gpu", "cpu", "directml"):
        raise ValidationError(
            "Unsupported runtime target",
            code="install.target_invalid",
            details={"target": body.target}, http_status=400,
        )
    try:
        res = onnxruntime_setup.install_runtime(body.target)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    stdout = res.pop("stdout", "")
    tail = "\n".join(stdout.splitlines()[-30:])
    # 同时返回当前进程视角（providers 仍是旧的，UI 用来对比）
    rt = onnxruntime_setup.current_runtime()
    return {
        **res,
        **rt,
        "cuda_detect": onnxruntime_setup.detect_cuda(),
        "accelerator": accelerator_block(),
        "stdout_tail": tail,
    }


# PyTorch 运行时 / 重装（PR-S2）-------------------------------------------


@router.get("/api/torch/status")
def torch_status() -> dict[str, Any]:
    """返回 torch 当前状态 + 驱动检测 + 推荐 cu tag + 误装诊断 flag。

    UI 用 `is_cpu_with_gpu` 决定是否显著提示「检测到 GPU 但装的是 CPU 版」。
    `is_cuda_build_unavailable` 标志驱动 / WSL 问题（不是 pip 能修的，UI 给文档链接）。

    `accelerator` 块在这里尤其关键：DCU 上 `capabilities.manage_torch_install=False`，
    前端必须据此置灰「重装 PyTorch」—— DTK torch 由镜像预装，pip 覆盖会直接报废
    环境且装不回来（见 utils.accelerator.should_manage_torch_install）。
    torch service 自己已经会带一份 `accelerator`（AcceleratorInfo.as_dict()）以及
    `can_manage_torch_install` / `manage_disabled_reason` / `is_backend_unavailable`。
    这里只把 `capabilities` 子对象补进去（`merge_accelerator_block`），让四个域的
    `accelerator` **形状一致** —— 前端一个 TS 类型吃全部端点，不必按端点分叉。
    """
    return merge_accelerator_block(torch_setup.current_status())


@router.post("/api/torch/reinstall")
def torch_reinstall(body: TorchReinstallRequest) -> dict[str, Any]:
    """注册 torch 重装请求；下次 Studio 启动时由 launcher 进程执行。

    为什么不直接装：server 进程已 import 了 torch（flash_attention_setup 等间接拉
    上的），Windows 上 `torch\\_C.cp311-win_amd64.pyd` 被锁，pip uninstall / replace
    会撞 [WinError 5] 拒绝访问。改成写 marker → 用户 Ctrl+C 重启 → cli.py 启动
    时还没 import torch，pip 能正常替换文件。

    返回 `{pending: true, target, tag, message}`，UI 显示「请关闭并重启 Studio」。
    """
    try:
        tag = torch_setup._decide_target_tag(body.target)
    except ValueError as exc:
        raise ValidationError(
            "Unsupported runtime target",
            code="install.target_invalid",
            details={"target": body.target}, http_status=400,
        ) from exc
    pending_install.register_torch_reinstall(body.target)
    return {
        "pending": True,
        "target": body.target,
        "tag": tag,
        "message": "重装请求已注册。请 Ctrl+C 关闭 Studio 后重新运行 studio.bat / studio.sh —— 启动时会自动安装 torch（~3 GB，5-30 分钟），然后正常起 server。",
    }


# FlashAttention runtime（PR-7b）-----------------------------------------


@router.get("/api/flash-attention/status")
def flash_attn_status() -> dict[str, Any]:
    """返回 flash_attn 安装状态 + 当前环境检测 + GitHub 候选 wheel 列表。

    candidates 里 score / tags 等 UI 不需要的字段已剥掉，只保留 url/name/notes/usable。
    候选最多取前 20 个，避免 GitHub 历史 release 一大坨刷屏。
    fetch_error 非 None 表示 GitHub API 请求失败（限流 / 网络 / 国内防火墙）；
    UI 要展示这条让用户能选择手动粘 URL。

    任何意外异常都包成 fetch_error 返回 200 —— 这是 Settings 页 mount 就拉的诊断
    数据，宁可降级显示「无法拉候选」也不要 500 把整段 UI 打成「加载失败」让用户
    误以为后端坏了。真出问题靠 server log 里的 traceback 排查。

    DCU 上 `env.supports_prebuilt_wheels=False`、candidates 恒为空且 fetch_error
    为 null（不是拉取失败，UI 不该显示网络排错文案）；前端据此显示「此后端不支持
    自动安装，需从 DTK 渠道装」。
    """
    try:
        status = flash_attention_setup.current_status()
        env = flash_attention_setup.detect_env()
        candidates, fetch_error = flash_attention_setup.find_candidates(env)
        slim = [
            {"url": c["url"], "name": c["name"], "notes": c["notes"], "usable": c["usable"]}
            for c in candidates[:20]
        ]
        return {
            **status,
            "env": env,
            "candidates": slim,
            "fetch_error": fetch_error,
            "accelerator": accelerator_block(),
        }
    except Exception as exc:  # noqa: BLE001
        # logger.exception 把 traceback 落盘 + 带 trace_id（trace middleware bound）
        import logging  # noqa: PLC0415
        logging.getLogger(__name__).exception("flash_attn status endpoint failed")
        return {
            "installed": False,
            "version": None,
            "env": {
                "python_tag": None,
                "cuda_tag": None,
                "cuda_ver": None,
                "driver_cuda_ver": None,
                "torch_tag": None,
                "torch_ver": None,
                "torch_cuda_build": None,
                "platform": None,
                "backend": None,
                "vendor_label": None,
                "hip_ver": None,
                # 兜底给 True：诊断挂掉时不该顺手把安装按钮永久置灰，用户至少还能
                # 手动粘 URL 自救；真在 DCU 上点了，service 层的拦截会给明确报错。
                "supports_prebuilt_wheels": True,
            },
            "candidates": [],
            "fetch_error": f"诊断失败：{type(exc).__name__}: {exc}",
            "accelerator": accelerator_block(),
        }


@router.post("/api/flash-attention/install")
def flash_attn_install(body: FlashAttnInstallRequest) -> dict[str, Any]:
    """安装 flash_attn wheel；url=null 走 service 的自动匹配。

    同步 pip install（远端 wheel ~150MB），可能几分钟；UI 按钮必须带 loading。
    flash_attn 是 C extension，装完必须重启 Studio 才能切换；返回 restart_required=True。
    """
    try:
        return flash_attention_setup.install(body.url)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc


# xformers runtime ------------------------------------------------------


@router.get("/api/xformers/status")
def xformers_status() -> dict[str, Any]:
    """返回 xformers 安装状态 + 后端能力。

    比 flash_attention/status 简洁很多 —— xformers 走 PyPI 直装，不需要 GitHub
    候选 wheel 列表 / 环境检测细节（status 里 installed/version 已经够用）。

    `accelerator.capabilities.xformers=False`（DCU）时前端置灰安装按钮并说明
    「DCU 走 SDPA 是正常路径」——不是能力缺失，见 xformers service 的 install()。
    """
    return {**xformers_setup.current_status(), "accelerator": accelerator_block()}


@router.post("/api/xformers/install")
def xformers_install() -> dict[str, Any]:
    """pip install xformers --index-url <torch-cu-index>。

    同步执行；远端 wheel 通常几十到几百 MB，几分钟级。装失败抛 500，message
    含 stderr 末尾（多数失败 = 上游 wheel 没覆盖当前 torch+cu 组合）。

    xformers 是 C extension，装完返回 restart_required=True 让 UI 提示重启。
    """
    try:
        return xformers_setup.install()
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc


# LLM tagger admin ------------------------------------------------------


def _select_preset(
    tagger_cfg: "secrets.LLMTaggerConfig", preset_id: Optional[str]
) -> "secrets.LLMPresetConfig":
    pid = preset_id or tagger_cfg.current_preset
    for preset in tagger_cfg.presets:
        if preset.id == pid:
            return preset
    return tagger_cfg.active


@router.post("/api/llm-tagger/models/refresh")
def refresh_llm_tagger_models(body: LLMModelsRefreshRequest) -> dict[str, Any]:
    """读取 OpenAI-compatible /models，并保存到指定 preset 的 model_ids。

    `preset_id` 不传时用 current_preset。成功后才落 secrets，避免请求失败时写脏。
    """
    from ...services.tagging import llm as llm_tagger_svc

    tagger_cfg = secrets.load().llm_tagger
    target = _select_preset(tagger_cfg, body.preset_id)
    base_url = (body.base_url if body.base_url is not None else target.base_url).strip()
    api_key = (
        target.api_key
        if body.api_key is None or body.api_key == secrets.MASK
        else body.api_key.strip()
    )
    if not base_url:
        raise ValidationError(
            "API base URL is required",
            code="llm_tagger.base_url_required", http_status=400,
        )
    try:
        model_ids = llm_tagger_svc.fetch_openai_compatible_models(
            base_url,
            api_key,
            timeout=body.timeout or target.timeout,
        )
    except Exception as exc:  # noqa: BLE001
        raise DomainError(
            f"Could not reach the model service: {exc}",
            code="llm_tagger.connect_failed",
            details={"reason": str(exc)}, http_status=502,
        ) from exc
    selected = target.model if target.model in model_ids else (model_ids[0] if model_ids else target.model)
    preset_patch: dict[str, Any] = {
        "id": target.id,
        "base_url": base_url,
        "model_ids": model_ids,
        "model": selected,
    }
    if body.api_key not in (None, secrets.MASK):
        preset_patch["api_key"] = api_key
    new = secrets.update({"llm_tagger": {"presets": [preset_patch]}})
    return {
        "items": model_ids,
        "preset_id": target.id,
        "secrets": secrets.to_masked_dict(new),
    }


@router.post("/api/llm-tagger/test")
def test_llm_tagger_connection(body: LLMConnectionTestRequest) -> dict[str, Any]:
    """Run a text-only LLM connectivity test without saving form values.

    Defaults come from the target preset (preset_id or current); body fields
    override on top.
    """
    from ...services.tagging import llm as llm_tagger_svc

    tagger_cfg = secrets.load().llm_tagger
    target = _select_preset(tagger_cfg, body.preset_id)
    merged = target.model_dump()
    for key in ("base_url", "model", "endpoint", "timeout", "max_tokens", "temperature"):
        value = getattr(body, key)
        if value is not None:
            merged[key] = value
    if body.api_key is not None and body.api_key != secrets.MASK:
        merged["api_key"] = body.api_key
    cfg = secrets.LLMPresetConfig(**merged)
    if not cfg.base_url.strip():
        raise ValidationError(
            "API base URL is required",
            code="llm_tagger.base_url_required", http_status=400,
        )
    if not cfg.model.strip():
        raise ValidationError(
            "A model must be selected",
            code="llm_tagger.model_required", http_status=400,
        )
    return llm_tagger_svc.test_openai_compatible_connection(
        cfg.base_url,
        cfg.api_key,
        cfg.model,
        endpoint=cfg.endpoint,
        timeout=cfg.timeout,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
