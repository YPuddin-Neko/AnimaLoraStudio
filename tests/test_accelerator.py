"""utils/accelerator.py — 后端识别 + 能力查询 + 显存采集（ADR 0016）。

这个模块是全仓库判断「NVIDIA CUDA / 海光 DCU / CPU」的单一权威源，被装包链路、
启动自检、显存护栏、topbar 监控共同消费。所以测试重点不是覆盖率，而是**误判的代价**：
把 DCU 判成 CPU 会让 Studio 推荐「重装 cu128 torch」，而那会覆盖掉 DTK 镜像预装的
torch、让整个容器报废（见 ADR 0016 背景段）。

真机无关：全部用假 torch 模块，不需要装 torch，也不需要有卡。
"""
from __future__ import annotations

import sys
import types

import pytest

from utils import accelerator


@pytest.fixture(autouse=True)
def _reset_cache():
    """清进程内后端缓存。

    `detect()` 刻意缓存（生产环境 torch build 在进程生命周期内不变），测试必须
    前后都清：前清保证本用例看到自己装的假 torch，后清保证不泄漏给下一个用例
    ——也不泄漏给本进程里**其他**测试文件（`_CACHE` 是模块级状态）。
    """
    accelerator._CACHE = None
    yield
    accelerator._CACHE = None


def _fake_torch(
    *, cuda=None, hip=None, version="2.9.0", available=True,
    device_names=("Fake Card",), gcn_arch=None,
):
    """造一个够用的假 torch 模块。

    **不用 `MagicMock()` 整体替身**：MagicMock 的属性访问会自动造 truthy 子 mock，
    `torch.version.hip` 于是永远为真 → 一切都被判成 DCU。这个坑真实踩过
    （`tests/test_torch_setup.py` 的 fake 必须显式写 `version.hip = None`），
    所以这里用真 module 对象 + 显式赋值，把两个版本字段都摆明。
    """
    mod = types.ModuleType("torch")
    mod.__version__ = version
    mod.version = types.SimpleNamespace(cuda=cuda, hip=hip)
    # SDPA 探测会 `torch.randn(..., dtype=torch.bfloat16)` 造探测张量。缺这两个属性
    # 时 AttributeError 被 probe 的兜底 except 吞掉、静默返回全 False —— 排查时很难
    # 看出是替身不全而不是被测逻辑错了，所以补齐并留此注释。
    mod.bfloat16 = "bfloat16"
    mod.randn = lambda *a, **k: object()

    props = [
        types.SimpleNamespace(
            total_memory=8 * 1024**3,
            **({"gcnArchName": gcn_arch} if gcn_arch else {}),
        )
        for _ in device_names
    ]
    mod.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: len(device_names),
        get_device_name=lambda i: device_names[i],
        get_device_properties=lambda i: props[i],
        mem_get_info=lambda i=0: (2 * 1024**3, 8 * 1024**3),
    )
    return mod


def _install(monkeypatch, mod):
    monkeypatch.setitem(sys.modules, "torch", mod)


# ---------------------------------------------------------------------------
# detect: 后端判定
# ---------------------------------------------------------------------------


def test_detect_nvidia_cuda_build(monkeypatch):
    _install(monkeypatch, _fake_torch(cuda="12.8", device_names=("RTX 5090",)))
    info = accelerator.detect()
    assert info.backend == "cuda"
    assert info.cuda_version == "12.8"
    assert info.hip_version is None
    assert info.device_names == ("RTX 5090",)
    assert info.is_gpu is True
    assert info.torch_device == "cuda"


def test_detect_dcu_hip_build(monkeypatch):
    """DTK wheel：version.cuda 为 None、version.hip 有值、版本串带 dtk 本地标签。

    这是本次移植的核心用例 —— 原实现按 `version.cuda is None` 判 CPU-only，
    正是在这里误判。
    """
    _install(monkeypatch, _fake_torch(
        hip="6.3.42134-abc", version="2.9.0+das.dtk2604",
        device_names=("Hygon BW1000",), gcn_arch="gfx928",
    ))
    info = accelerator.detect()
    assert info.backend == "dcu"
    assert info.cuda_version is None
    assert info.hip_version == "6.3.42134-abc"
    assert info.gcn_arch == ("gfx928",)
    assert info.is_gpu is True
    # DCU 上设备字符串**仍然**是 "cuda"（HIP 复用整套 CUDA 设备 API）
    assert info.torch_device == "cuda"
    assert info.vendor_label == "海光 DCU (DTK)"


def test_detect_cpu_only_build(monkeypatch):
    _install(monkeypatch, _fake_torch(version="2.9.0+cpu", available=False,
                                      device_names=()))
    info = accelerator.detect()
    assert info.backend == "cpu"
    assert info.is_gpu is False
    assert info.torch_device == "cpu"


def test_detect_hip_wins_when_both_present(monkeypatch):
    """两个版本字段都有值时判 DCU。

    正常 wheel 不会这样，但 ROCm build 历史上出现过 `version.cuda` 被填成
    HIP 兼容版本号的情况。判定顺序写死 hip 优先，这个用例把顺序锁住。
    """
    _install(monkeypatch, _fake_torch(cuda="12.8", hip="6.3.0"))
    assert accelerator.detect().backend == "dcu"


def test_detect_torch_missing_records_import_error(monkeypatch):
    """torch 没装 → backend=cpu 且 import_error 带原因。

    调用方要靠 `import_error` 区分「真 CPU 机器」与「torch 坏了」（后者在
    Windows 上表现为 DLL 加载失败的 OSError，不是 ImportError）。
    """
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def boom(name, *a, **k):
        if name == "torch":
            raise OSError("[WinError 126] 找不到指定的模块。")
        return real_import(name, *a, **k)

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr("builtins.__import__", boom)
    info = accelerator.detect()
    assert info.backend == "cpu"
    assert "WinError 126" in (info.import_error or "")


def test_detect_device_enumeration_failure_is_survivable(monkeypatch):
    """驱动问题让设备枚举抛错时，仍要给出 backend（只是 available=False）。

    容器没挂 /dev/kfd 就是这种状态：wheel 是 DTK 的（backend 判得出来），但
    `is_available()` / 枚举会失败。此时 cli.py 要能据 backend 给 DCU 专属排错提示，
    所以不能整体降级成 cpu。
    """
    mod = _fake_torch(hip="6.3.0")
    mod.cuda.is_available = lambda: (_ for _ in ()).throw(RuntimeError("no kfd"))
    _install(monkeypatch, mod)
    info = accelerator.detect()
    assert info.backend == "dcu"
    assert info.available is False
    assert info.device_count == 0


def test_detect_is_cached_until_refresh(monkeypatch):
    _install(monkeypatch, _fake_torch(cuda="12.8"))
    assert accelerator.detect().backend == "cuda"
    _install(monkeypatch, _fake_torch(hip="6.3.0"))
    assert accelerator.detect().backend == "cuda", "缓存失效了"
    assert accelerator.detect(refresh=True).backend == "dcu"


def test_as_dict_is_json_friendly(monkeypatch):
    """`as_dict()` 供 /api 与日志用，元组要转成 list。"""
    _install(monkeypatch, _fake_torch(hip="6.3.0", device_names=("A", "B"),
                                      gcn_arch="gfx928"))
    d = accelerator.detect().as_dict()
    assert isinstance(d["device_names"], list) and d["device_names"] == ["A", "B"]
    assert isinstance(d["gcn_arch"], list)
    assert d["vendor_label"] == "海光 DCU (DTK)"


# ---------------------------------------------------------------------------
# 能力查询
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs,backend", [
    ({"cuda": "12.8"}, "cuda"),
    ({"hip": "6.3.0"}, "dcu"),
    ({"available": False, "device_names": ()}, "cpu"),
])
def test_capability_matrix(monkeypatch, kwargs, backend):
    """能力矩阵一次性锁住 —— 上层全靠这几个函数决定装不装包。

    `should_manage_torch_install` 在 DCU 上为 False 是**最关键**的一条：
    它管住 Settings 重装按钮 / CLI --torch / studio.sh 首装三条路。
    """
    _install(monkeypatch, _fake_torch(**kwargs))
    assert accelerator.backend() == backend

    expected_cuda_only = backend == "cuda"
    assert accelerator.can_pip_install_xformers() is expected_cuda_only
    assert accelerator.supports_prebuilt_flash_attn_wheels() is expected_cuda_only
    assert accelerator.should_manage_torch_install() is (backend != "dcu")
    assert accelerator.is_dcu() is (backend == "dcu")
    assert accelerator.is_nvidia() is (backend == "cuda")


def test_xformers_availability_is_probed_not_inferred_from_backend(monkeypatch):
    """DCU 上 xformers **可用性**不由后端决定 —— 装了海光配套 wheel 就该报可用。

    这条钉住本次移植修正过的一个错误假设：最初把「DCU 不支持 xformers」写进能力
    矩阵，而海光在光合社区发布配套 wheel（xformers-0.0.33+das.opt1.dtk2604.torch251），
    装上后 NaViT 打包等功能照常可用。判据必须是实测，不是卡型号。
    """
    _install(monkeypatch, _fake_torch(hip="6.3.0"))
    # 装了且能跑 → 可用（即便后端是 DCU）
    ops = types.ModuleType("xformers.ops")
    ops.memory_efficient_attention = lambda *a, **k: "ok"
    xf = types.ModuleType("xformers")
    xf.ops = ops
    monkeypatch.setitem(sys.modules, "xformers", xf)
    monkeypatch.setitem(sys.modules, "xformers.ops", ops)
    accelerator._XFORMERS_CACHE = None
    assert accelerator.xformers_works() is True
    # 但「能否 pip 自动装」仍是 False —— 公开源上没有 DCU wheel
    assert accelerator.can_pip_install_xformers() is False


def test_xformers_works_false_when_kernel_raises(monkeypatch):
    """import 成功但 kernel 跑不起来 → 不可用。

    海光的 xformers wheel 是 py3-none-any（纯 Python），底层 kernel 依赖 DTK 侧实现，
    所以「import 成功」不等于「能算」，必须实测。
    """
    _install(monkeypatch, _fake_torch(hip="6.3.0"))
    ops = types.ModuleType("xformers.ops")

    def _boom(*a, **k):
        raise RuntimeError("No operator found for memory_efficient_attention_forward")

    ops.memory_efficient_attention = _boom
    xf = types.ModuleType("xformers")
    xf.ops = ops
    monkeypatch.setitem(sys.modules, "xformers", xf)
    monkeypatch.setitem(sys.modules, "xformers.ops", ops)
    accelerator._XFORMERS_CACHE = None
    assert accelerator.xformers_works() is False


@pytest.mark.parametrize("kwargs,expected", [
    ({"cuda": "12.8"}, "CUDAExecutionProvider"),
    ({"hip": "6.3.0"}, "MIGraphXExecutionProvider"),
    ({"available": False, "device_names": ()}, None),
])
def test_onnx_gpu_provider(monkeypatch, kwargs, expected):
    _install(monkeypatch, _fake_torch(**kwargs))
    assert accelerator.onnx_gpu_provider() == expected


# ---------------------------------------------------------------------------
# probe_stdlib: 不依赖 torch 的硬件探测（bootstrap 阶段用）
# ---------------------------------------------------------------------------


def test_probe_stdlib_detects_nvidia(monkeypatch):
    monkeypatch.setattr(accelerator.shutil, "which",
                        lambda c: "/usr/bin/nvidia-smi" if c == "nvidia-smi" else None)
    monkeypatch.setattr(accelerator, "_run",
                        lambda args, timeout=10: "550.54.15, NVIDIA GeForce RTX 4090")
    p = accelerator.probe_stdlib()
    assert p.backend == "cuda"
    assert p.driver_version == "550.54.15"
    assert p.gpu_name == "NVIDIA GeForce RTX 4090"


def test_probe_stdlib_detects_dcu_via_hy_smi(monkeypatch):
    """nvidia-smi 探测失败 + PATH 上有 hy-smi → DCU。"""
    monkeypatch.setattr(accelerator.shutil, "which",
                        lambda c: "/opt/hyhal/bin/hy-smi" if c == "hy-smi" else None)

    def fake_run(args, timeout=10):
        if args[0] == "nvidia-smi":
            return None
        if "--version" in args:
            return "HY-SMI version: 1.4.1"
        return "GPU[0]\t: Card series: BW1000"

    monkeypatch.setattr(accelerator, "_run", fake_run)
    p = accelerator.probe_stdlib()
    assert p.backend == "dcu"
    assert p.driver_version == "1.4.1"
    assert p.gpu_name == "BW1000"


def test_probe_stdlib_dcu_via_dev_kfd_without_smi(monkeypatch):
    """没有任何 smi 工具但 /dev/kfd 在 → 仍判 DCU。

    DTK 装得不全时会是这样；torch 侧大概率照样能用，所以不能因为缺 smi 就
    判成 cpu（那会让 bootstrap 去装 PyPI 的 CPU torch，覆盖掉预装的 DTK torch）。
    """
    monkeypatch.setattr(accelerator.shutil, "which", lambda c: None)
    monkeypatch.setattr(accelerator, "_run", lambda args, timeout=10: None)
    monkeypatch.setattr(accelerator.os.path, "exists", lambda p: p == "/dev/kfd")
    p = accelerator.probe_stdlib()
    assert p.backend == "dcu"
    assert p.driver_version is None


def test_probe_stdlib_cpu_when_nothing_found(monkeypatch):
    monkeypatch.setattr(accelerator.shutil, "which", lambda c: None)
    monkeypatch.setattr(accelerator, "_run", lambda args, timeout=10: None)
    monkeypatch.setattr(accelerator.os.path, "exists", lambda p: False)
    assert accelerator.probe_stdlib().backend == "cpu"


@pytest.mark.parametrize("text,expected", [
    ("HY-SMI version: 1.4.1", "1.4.1"),
    ("ROCM-SMI version: 1.4.1\nROCM-SMI-LIB version: 5.0.0", "1.4.1"),
    ("", None),
    ("no digits here", None),
])
def test_parse_dcu_driver_version(text, expected):
    assert accelerator._parse_dcu_driver_version(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("GPU[0]\t: Card series: BW1000", "BW1000"),
    ("Card series:\tZ100L", "Z100L"),
    ("device: K100_AI", "K100_AI"),
    ("nothing recognizable", None),
])
def test_parse_dcu_gpu_name(text, expected):
    """型号解析是 best-effort（权威型号名来自 torch），但不能误报。"""
    assert accelerator._parse_dcu_gpu_name(text) == expected


# ---------------------------------------------------------------------------
# 显存查询：按后端选路径
# ---------------------------------------------------------------------------


def _fake_pynvml(monkeypatch, *, free_bytes=None, devices=None, init_raises=False):
    """装一个假 pynvml。`devices` 为 (name, used, total, util, temp) 元组列表。"""
    mod = types.ModuleType("pynvml")
    mod.NVML_TEMPERATURE_GPU = 0

    def _init():
        if init_raises:
            raise RuntimeError("NVML init failed (no NVIDIA driver)")

    mod.nvmlInit = _init
    mod.nvmlShutdown = lambda: None
    mod.nvmlDeviceGetHandleByIndex = lambda i: i
    if free_bytes is not None:
        mod.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(
            free=free_bytes, used=1, total=2,
        )
    if devices is not None:
        mod.nvmlDeviceGetCount = lambda: len(devices)
        mod.nvmlDeviceGetName = lambda h: devices[h][0]
        mod.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(
            used=devices[h][1], total=devices[h][2], free=devices[h][2] - devices[h][1],
        )
        mod.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(
            gpu=devices[h][3],
        )
        mod.nvmlDeviceGetTemperature = lambda h, t: devices[h][4]
    monkeypatch.setitem(sys.modules, "pynvml", mod)
    return mod


def test_free_vram_prefers_nvml_on_nvidia(monkeypatch):
    """NVIDIA 上**必须**优先 NVML：WDDM 下 mem_get_info 是每进程视角，
    看不到他进程占用，拿它做跨进程护栏形同虚设（ADR 0016 / sysmem.py 踩坑记录）。

    假 NVML 报 7GB、假 torch 报 2GB —— 断言拿到的是 NVML 那个数。
    """
    _install(monkeypatch, _fake_torch(cuda="12.8"))
    _fake_pynvml(monkeypatch, free_bytes=7 * 1024**3)
    assert accelerator.free_vram_bytes() == 7 * 1024**3


def test_free_vram_uses_torch_on_dcu_even_if_pynvml_importable(monkeypatch):
    """DCU 上不走 NVML —— 即使 pynvml 装着（纯 Python 包，装得上）。

    真机上它 init 会失败，但「装着且能 import」在 DCU 上完全可能（requirements.txt
    里就有 nvidia-ml-py）。所以路径选择必须靠 backend 判定，不能靠 import 成功与否。
    这里刻意让假 NVML 能 init 并报一个不同的数，确认它没被采纳。
    """
    _install(monkeypatch, _fake_torch(hip="6.3.0"))
    _fake_pynvml(monkeypatch, free_bytes=7 * 1024**3)
    assert accelerator.free_vram_bytes() == 2 * 1024**3  # 来自 fake torch


def test_free_vram_falls_back_to_torch_when_nvml_init_fails(monkeypatch):
    _install(monkeypatch, _fake_torch(cuda="12.8"))
    _fake_pynvml(monkeypatch, init_raises=True)
    assert accelerator.free_vram_bytes() == 2 * 1024**3


def test_free_vram_none_when_nothing_available(monkeypatch):
    _install(monkeypatch, _fake_torch(available=False, device_names=()))
    monkeypatch.setitem(sys.modules, "pynvml", types.ModuleType("pynvml"))
    assert accelerator.free_vram_bytes() is None


def test_device_stats_nvml_path_has_util_and_temp(monkeypatch):
    _install(monkeypatch, _fake_torch(cuda="12.8"))
    _fake_pynvml(monkeypatch, devices=[("RTX 4090", 4 * 1024**3, 24 * 1024**3, 67, 50)])
    stats = accelerator.device_stats()
    assert stats is not None and len(stats) == 1
    d = stats[0]
    assert (d.name, d.util_pct, d.temp_c) == ("RTX 4090", 67, 50)
    assert (d.vram_used_gb, d.vram_total_gb) == (4.0, 24.0)


def test_device_stats_nvml_decodes_bytes_name(monkeypatch):
    """老 NVML 返回 bytes 名字；不解码会让前端显示 b'...'。"""
    _install(monkeypatch, _fake_torch(cuda="12.8"))
    _fake_pynvml(monkeypatch, devices=[(b"RTX 3090", 1024**3, 24 * 1024**3, 10, 40)])
    stats = accelerator.device_stats()
    assert stats is not None and stats[0].name == "RTX 3090"


def test_device_stats_torch_path_reports_vram_without_smi(monkeypatch):
    """DCU 口径：smi 不可用时显存照常返回（绝对值来自 torch），两项指标留 None。

    ⚠️ ``smi_command`` 必须显式 stub。这条原来只 stub 了 torch，于是在**真机上失败、
    在没有 hy-smi 的机器上通过** —— 正好反了：真机 DCU 上 ``_dcu_smi_metrics`` 会真
    去跑 hy-smi 并成功返回 ``util_pct=0, temp_c=51``，而断言写的是两项为 None。那是
    我后来加 ``_parse_hy_smi_metrics`` 之前的行为，加了解析后没回来改这条测试。

    教训：凡是会 shell out 的依赖，测试里必须显式 stub。靠"目标环境大概没装这个工具"
    让断言成立，等于把结果绑在环境上 —— 而这类测试最需要在真机上跑。
    """
    _install(monkeypatch, _fake_torch(hip="6.3.0", device_names=("Hygon BW1000",)))
    monkeypatch.setattr(accelerator, "smi_command", lambda: None)
    stats = accelerator.device_stats()
    assert stats is not None and len(stats) == 1
    d = stats[0]
    assert d.name == "Hygon BW1000"
    assert d.util_pct is None and d.temp_c is None
    assert (d.vram_used_gb, d.vram_total_gb) == (6.0, 8.0)  # (total-free)/1G, total/1G


def test_device_stats_smi_partial_coverage_leaves_missing_cards_none(monkeypatch):
    """smi 只报了部分卡时，缺的那些卡两项为 None，不能串到别的卡上。

    真机 hy-smi 偶发只列出部分卡（驱动重载 / 卡被占用时）。``_torch_device_stats``
    用 ``metrics.get(i, (None, None))`` 按 index 取而不是按顺序 zip，就是为了这种
    情形 —— 顺序 zip 会把 1 号卡的读数安到 0 号卡上。
    """
    _install(
        monkeypatch,
        _fake_torch(hip="6.3.0", device_names=("Hygon BW1000", "Hygon BW1000")),
    )
    monkeypatch.setattr(accelerator, "smi_command", lambda: "/opt/hyhal/bin/hy-smi")
    monkeypatch.setattr(accelerator, "_parse_hy_smi_metrics", lambda _text: {1: (12, 55)})
    monkeypatch.setattr(accelerator, "_run", lambda args, timeout=10: "irrelevant")
    stats = accelerator.device_stats()
    assert stats is not None and len(stats) == 2
    assert (stats[0].util_pct, stats[0].temp_c) == (None, None), "0 号卡不该有读数"
    assert (stats[1].util_pct, stats[1].temp_c) == (12, 55), "1 号卡的读数错位了"


def test_device_stats_none_without_gpu(monkeypatch):
    """无加速器 → None（调用方据此隐藏 GPU pill，而不是显示 0 卡）。"""
    _install(monkeypatch, _fake_torch(available=False, device_names=()))
    assert accelerator.device_stats() is None


# ---------------------------------------------------------------------------
# SDPA 后端探测与配置（ADR 0016 / 真机 DTK 26.04 实测驱动的修复）
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_probe_caches():
    """清 SDPA / xformers 探测缓存。与 `_CACHE` 同理：模块级、跨调用保持，
    测试间不清会串味（前一个用例的假 torch 结论泄漏给后一个）。"""
    accelerator._SDPA_CACHE = None
    accelerator._XFORMERS_CACHE = None
    yield
    accelerator._SDPA_CACHE = None
    accelerator._XFORMERS_CACHE = None


def _install_sdpa_torch(monkeypatch, *, works, hip=None, cuda=None, available=True):
    """装一个能模拟「按后端成功/失败」的假 torch。

    `works` 是后端名集合（``{"flash", "mem_efficient", "math"}`` 的子集）；
    `sdpa_kernel(b)` 上下文里调 SDPA 时，不在集合里的后端抛 RuntimeError。
    无上下文的裸调用（= SDPA 默认 dispatch）模拟 torch 真实行为：**先试 flash**，
    flash 不可用就抛 —— 不会优雅回落。这正是真机上 `sdpa_default` 也失败的原因。
    """
    mod = _fake_torch(hip=hip, cuda=cuda, available=available)

    current = {"backend": None}

    class _Ctx:
        def __init__(self, backend):
            self._backend = backend

        def __enter__(self):
            current["backend"] = self._backend
            return self

        def __exit__(self, *a):
            current["backend"] = None
            return False

    class _SDPBackend:
        FLASH_ATTENTION = "flash"
        EFFICIENT_ATTENTION = "mem_efficient"
        MATH = "math"
        CUDNN_ATTENTION = "cudnn"

    def _sdpa(*_a, **_k):
        # 无 ctx = 默认 dispatch；torch 会先试 flash，失败即抛
        target = current["backend"] or "flash"
        if target not in works:
            raise RuntimeError(
                f"No matching libraries found for flash_attn_2_cuda*.so ({target})"
            )
        return "ok"

    flags = {"flash": True, "mem_efficient": True, "math": True, "cudnn": True}
    mod.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            enable_flash_sdp=lambda v: flags.__setitem__("flash", v),
            enable_mem_efficient_sdp=lambda v: flags.__setitem__("mem_efficient", v),
            enable_math_sdp=lambda v: flags.__setitem__("math", v),
            enable_cudnn_sdp=lambda v: flags.__setitem__("cudnn", v),
        ),
    )
    # `import torch.nn.functional as F` 要求**整条 submodule 链**都在 sys.modules
    # 里，而且父模块得有对应属性 —— 只塞叶子节点会走进 except 返回全 False。
    functional = types.ModuleType("torch.nn.functional")
    functional.scaled_dot_product_attention = _sdpa
    attention = types.ModuleType("torch.nn.attention")
    attention.SDPBackend = _SDPBackend
    attention.sdpa_kernel = _Ctx
    nn = types.ModuleType("torch.nn")
    nn.functional = functional
    nn.attention = attention
    mod.nn = nn

    _install(monkeypatch, mod)
    monkeypatch.setitem(sys.modules, "torch.nn", nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", functional)
    monkeypatch.setitem(sys.modules, "torch.nn.attention", attention)
    return flags


def test_probe_sdpa_matches_real_dcu_result(monkeypatch):
    """复刻真机结果：只有 math 可用，**连默认 dispatch 也失败**。

    真机（BW / gfx936 / DTK 26.04 / torch 2.9.0）实测：
        sdpa_default / sdpa_flash → RuntimeError: No matching libraries ...
        sdpa_mem_efficient        → RuntimeError: No available kernel.
        sdpa_math                 → OK
    `default` 也失败是整个修复的**理由** —— 否则「不配置、让 SDPA 自己选」就够了。
    """
    _install_sdpa_torch(monkeypatch, works={"math"}, hip="6.3.26093")
    probed = accelerator.probe_sdpa_backends()
    assert probed == {
        "flash": False, "mem_efficient": False, "math": True, "default": False,
    }


def test_configure_sdpa_disables_broken_backends_on_dcu(monkeypatch):
    """DCU + 只有 math 可用 → 关掉 flash / mem_efficient / cudnn，math 保持开。

    关掉是为了让 SDPA 的 dispatch 根本不去试它们；否则第一个 attention 调用就抛。
    """
    flags = _install_sdpa_torch(monkeypatch, works={"math"}, hip="6.3.26093")
    accelerator.configure_sdpa()
    assert flags["flash"] is False
    assert flags["mem_efficient"] is False
    assert flags["cudnn"] is False
    assert flags["math"] is True


def test_configure_sdpa_keeps_working_backends_on_dcu(monkeypatch, caplog):
    """装上海光 flash-attn 后的真机终态：flash + math 可用、mem_efficient 不可用。

    这就是 BW1000 / DTK 26.04 / torch 2.5.1 + flash-attn 2.6.1 的实测结果。两个要点：
    1. flash 探测通过 → **不关它**，自动走快路径（用户装包即生效，无需改配置）
    2. 只缺 mem_efficient 时**不该 warn** —— 这是该平台的正常终态（DTK 编译时未开
       该后端），flash 更快本就优先它，每次启动刷警告只是噪声
    """
    flags = _install_sdpa_torch(
        monkeypatch, works={"flash", "math"}, hip="6.3.26093",
    )
    with caplog.at_level("WARNING"):
        accelerator.configure_sdpa()
    assert flags["flash"] is True
    assert flags["math"] is True
    assert flags["mem_efficient"] is False
    assert not caplog.records, f"flash 可用时不该有警告，实际: {caplog.text}"


def test_configure_sdpa_warns_only_when_flash_unavailable(monkeypatch, caplog):
    """flash 不可用才 warn，且文案要给出可执行的下一步（装包 / 查本机）。"""
    _install_sdpa_torch(monkeypatch, works={"math"}, hip="6.3.26093")
    with caplog.at_level("WARNING"):
        accelerator.configure_sdpa()
    assert caplog.records, "flash 不可用时必须警告"
    assert "flash-attn" in caplog.text
    assert "find_flash_attn.sh" in caplog.text


def test_configure_sdpa_touches_nothing_on_nvidia(monkeypatch):
    """NVIDIA 上只探测、**不动任何开关** —— 本次移植不许改既有 NVIDIA 行为。

    刻意用一个「flash 不可用」的假 NVIDIA 环境：即便探测说不行，也不该去关开关
    （NVIDIA 上 SDPA 的 dispatch 与回落一向正常，关开关只会引入新行为）。
    """
    flags = _install_sdpa_torch(monkeypatch, works={"math"}, cuda="12.8")
    accelerator.configure_sdpa()
    assert flags == {"flash": True, "mem_efficient": True, "math": True, "cudnn": True}


def test_usable_sdpa_backends_orders_fast_first(monkeypatch):
    """返回实测可用的后端，快的在前（flash → mem_efficient → math）。"""
    _install_sdpa_torch(
        monkeypatch, works={"flash", "mem_efficient", "math"}, hip="6.3.0",
    )
    assert accelerator.usable_sdpa_backends() == ["flash", "mem_efficient", "math"]


def test_usable_sdpa_backends_excludes_broken(monkeypatch):
    """DCU 现状：列表里只剩 math。

    `comfy_qwen.py` 用这个列表喂 `sdpa_kernel(priority, set_priority=True)` ——
    那种写法会覆盖全局开关，所以列表里绝不能混进不可用的后端。
    """
    _install_sdpa_torch(monkeypatch, works={"math"}, hip="6.3.0")
    assert accelerator.usable_sdpa_backends() == ["math"]


def test_usable_sdpa_backends_falls_back_to_math_when_all_fail(monkeypatch):
    """极端情况全失败 → 仍返回 [MATH] 而非空列表 / None。

    空列表喂给 `sdpa_kernel` 会 ValueError，把一个「慢」的问题变成「崩」的问题；
    返回 MATH 让报错（如果真有）发生在 SDPA 内部，信息更有用。
    """
    _install_sdpa_torch(monkeypatch, works=set(), hip="6.3.0")
    assert accelerator.usable_sdpa_backends() == ["math"]


def test_probe_sdpa_no_gpu_reports_all_false(monkeypatch):
    _install_sdpa_torch(monkeypatch, works={"math"}, available=False)
    probed = accelerator.probe_sdpa_backends()
    assert probed == {
        "flash": False, "mem_efficient": False, "math": False, "default": False,
    }


def test_probe_sdpa_is_cached(monkeypatch):
    """探测要真跑 SDPA，不能每次调用都付这个代价。"""
    calls = [0]
    _install_sdpa_torch(monkeypatch, works={"math"}, hip="6.3.0")
    real = accelerator.probe_sdpa_backends

    monkeypatch.setattr(
        sys.modules["torch.nn.functional"], "scaled_dot_product_attention",
        lambda *a, **k: (calls.__setitem__(0, calls[0] + 1), "ok")[1],
    )
    real()
    first = calls[0]
    real()
    assert calls[0] == first, "探测结果未缓存"


# ---------------------------------------------------------------------------
# hy-smi 指标解析（真机格式，DTK 26.04 / BW1000）
# ---------------------------------------------------------------------------

#: 真机 `hy-smi` 原样输出（2 卡空载）。写死在测试里是刻意的 —— 这是从目标环境
#: 采集到的事实基线，parser 改动不能悄悄破坏对它的解析能力。
_REAL_HY_SMI = """\
================================= System Management Interface ==================================
================================================================================================
HCU     Temp     AvgPwr     Perf     PwrCap     VRAM%      HCU%      Dec%      Enc%      Mode
0       50.0C    80.0W      auto     1000.0W    0%         0.0%      0.0%      0.0%      Normal
1       52.0C    89.0W      auto     1000.0W    0%         0.0%      0.0%      0.0%      Normal
================================================================================================
======================================== End of SMI Log ========================================"""


def test_parse_hy_smi_real_output():
    assert accelerator._parse_hy_smi_metrics(_REAL_HY_SMI) == {0: (0, 50), 1: (0, 52)}


def test_parse_hy_smi_busy_card_rounds():
    """非零负载：利用率与温度都按四舍五入取整（97.5% → 98）。"""
    row = "0       78.5C    310.0W     auto     1000.0W    64%        97.5%     0.0%      0.0%      Normal"
    assert accelerator._parse_hy_smi_metrics(row) == {0: (98, 78)}


@pytest.mark.parametrize("text", [
    "",
    "HCU     Temp     AvgPwr     Perf     PwrCap     VRAM%      HCU%      Dec%      Enc%      Mode",
    "some unrelated log line\n=== End of SMI Log ===",
    "0  garbage  columns  here",
])
def test_parse_hy_smi_non_data_lines_yield_nothing(text):
    """表头 / 分隔线 / 无关日志都不该被当成数据行。

    parser 是 best-effort（这两项是 topbar 可选渲染），但**不能误报** —— 把表头
    解析成一张卡会让前端显示一个不存在的设备。
    """
    assert accelerator._parse_hy_smi_metrics(text) == {}


def test_torch_device_stats_fills_util_temp_from_smi_on_dcu(monkeypatch):
    """DCU 路径：显存来自 torch，利用率 / 温度来自 smi。

    分工的理由见 `_torch_device_stats` docstring —— smi 只给 VRAM%（百分比），
    而 topbar 要绝对值；torch 又拿不到利用率与温度。
    """
    _install(monkeypatch, _fake_torch(hip="6.3.0", device_names=("BW", "BW")))
    monkeypatch.setattr(accelerator, "smi_command", lambda: "/opt/hyhal/bin/hy-smi")
    monkeypatch.setattr(accelerator, "_run", lambda args, timeout=10: _REAL_HY_SMI)
    stats = accelerator.device_stats()
    assert stats is not None and len(stats) == 2
    assert (stats[0].util_pct, stats[0].temp_c) == (0, 50)
    assert (stats[1].util_pct, stats[1].temp_c) == (0, 52)
    # 显存仍来自 torch（fake 报 free=2G / total=8G），不是 smi 的 VRAM%
    assert stats[0].vram_total_gb == 8.0


def test_torch_device_stats_survives_smi_missing(monkeypatch):
    """smi 不存在时显存照常返回，两项指标留 None（前端隐藏那两个 pill）。"""
    _install(monkeypatch, _fake_torch(hip="6.3.0", device_names=("BW",)))
    monkeypatch.setattr(accelerator, "smi_command", lambda: None)
    stats = accelerator.device_stats()
    assert stats is not None
    assert stats[0].util_pct is None and stats[0].temp_c is None
    assert stats[0].vram_total_gb == 8.0


def test_smi_command_uses_backend_candidates(monkeypatch):
    """smi 工具按后端选：DCU 优先 hy-smi，退 rocm-smi。"""
    _install(monkeypatch, _fake_torch(hip="6.3.0"))
    monkeypatch.setattr(accelerator.shutil, "which",
                        lambda c: "/opt/rocm/bin/rocm-smi" if c == "rocm-smi" else None)
    assert accelerator.smi_command() == "/opt/rocm/bin/rocm-smi"

    _install(monkeypatch, _fake_torch(cuda="12.8"))
    accelerator._CACHE = None
    monkeypatch.setattr(accelerator.shutil, "which",
                        lambda c: "/opt/rocm/bin/rocm-smi" if c == "rocm-smi" else None)
    assert accelerator.smi_command() is None, "NVIDIA 后端不该挑 rocm-smi"

# ---------------------------------------------------------------------------
# 测试自身的卫生：不许把结果绑在运行环境上
# ---------------------------------------------------------------------------


def test_no_dcu_test_reaches_the_real_smi_binary() -> None:
    """凡是走 DCU 路径又调 ``device_stats`` 的测试，都必须 stub 掉 shell 依赖。

    守的是一类真实踩过的坑，而不是某一行：
    ``test_device_stats_torch_path_leaves_util_temp_none`` 原来只 stub 了 torch，
    ``_dcu_smi_metrics`` 于是真去跑 hy-smi。后果是这条测试**在真机 DCU 上失败、在
    没装 hy-smi 的开发机上通过** —— 完全反了。而 DCU 相关的改动最需要在真机上验，
    这种测试等于在最该起作用的地方失效。

    判定方式：函数体里出现 ``hip=`` （走 DCU 分支）且调用了 ``device_stats``，就必须
    同时 stub 掉 ``smi_command`` / ``_dcu_smi_metrics`` 之一。stub 哪一层都行 ——
    ``smi_command`` 更深、顺带覆盖解析，``_dcu_smi_metrics`` 更省事。

    为什么用 AST 而不是 conftest 里禁 subprocess：``_run`` 的失败会被 best-effort 的
    except 吞掉，禁用它只会让测试静默地测到兜底分支，看起来仍然是绿的。
    """
    import ast
    import pathlib

    source = pathlib.Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    #: 允许 stub 的目标：任一即可。
    SMI_STUBS = {"smi_command", "_dcu_smi_metrics", "_parse_hy_smi_metrics", "_run"}

    def _calls_and_strings(fn: ast.FunctionDef) -> tuple[set[str], set[str]]:
        """``(被调用的函数名, monkeypatch.setattr 的目标名)``。

        **不看 ast.dump 的整体文本**：那会把 docstring 里的散文也算成命中 —— 本函数
        的 docstring 就提到 ``smi_command``，早先的版本因此对自己的反例视而不见。
        只认真正的调用节点与 setattr 的字符串实参。
        """
        called: set[str] = set()
        patched: set[str] = set()
        for sub in ast.walk(fn):
            if not isinstance(sub, ast.Call):
                continue
            func = sub.func
            if isinstance(func, ast.Attribute):
                called.add(func.attr)
                if func.attr == "setattr":
                    for arg in sub.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            patched.add(arg.value)
            elif isinstance(func, ast.Name):
                called.add(func.id)
        return called, patched

    def _uses_dcu(fn: ast.FunctionDef) -> bool:
        """函数里是否用 ``hip=`` 造过 DCU 替身（只看关键字实参，不看散文）。"""
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Call) and any(
                kw.arg == "hip" and not (
                    isinstance(kw.value, ast.Constant) and kw.value.value is None
                )
                for kw in sub.keywords
            ):
                return True
        return False

    offenders = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        called, patched = _calls_and_strings(node)
        if "device_stats" not in called:
            continue
        if not _uses_dcu(node):            # 非 DCU 路径（NVML / 无卡）不受这条约束
            continue
        if patched & SMI_STUBS:
            continue
        offenders.append(node.name)

    assert not offenders, (
        "下列 DCU 测试调了 device_stats 却没 stub smi 依赖，会在真机上真跑 hy-smi ——\n"
        + "\n".join(f"  {name}" for name in offenders)
        + "\n加一行 monkeypatch.setattr(accelerator, 'smi_command', lambda: None) "
        "（或按需返回假读数）。"
    )
