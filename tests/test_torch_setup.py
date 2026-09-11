"""PR-S2 — torch_setup 服务：detect + recommend + reinstall。"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from unittest.mock import MagicMock

import pytest

from studio.services.runtime import torch as ts
from utils import accelerator


@pytest.fixture(autouse=True)
def _reset_accelerator_cache():
    """每个用例前后清 `utils.accelerator` 的进程内后端缓存（ADR 0019）。

    `detect()` 刻意缓存结果 —— 生产环境里 torch build 在进程生命周期内不会变。
    但本文件的用例靠 `monkeypatch.setitem(sys.modules, "torch", fake)` 换掉 torch，
    如果缓存已被别的用例（或 import 期的真实探测）填好，`detect()` 就看不见 fake，
    `cuda_build` 会全部退化成宿主机的真实后端。前后都清是因为：前清保证本用例看到
    fake，后清保证本用例的 fake 不泄漏给下一个用例。
    """
    accelerator._CACHE = None
    yield
    accelerator._CACHE = None


# ---------------------------------------------------------------------------
# recommend_cu_tag: 驱动版本 → cu wheel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver,expected", [
    ("570.15", "cu128"),  # >= 555 → cu128
    ("555.86", "cu128"),  # 边界
    ("550.10", "cu126"),  # 边界
    ("547.0", "cu124"),
    ("530.30", "cu118"),  # >= 470
    ("470.0", "cu118"),   # 边界
    ("460.50", "cpu"),    # 太老
    (None, "cpu"),
    ("", "cpu"),
    ("not-a-version", "cpu"),
])
def test_recommend_cu_tag(driver, expected) -> None:
    assert ts.recommend_cu_tag(driver) == expected


# ---------------------------------------------------------------------------
# detect_torch
# ---------------------------------------------------------------------------


def test_detect_torch_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_pkg):
        raise PackageNotFoundError
    monkeypatch.setattr(ts, "_pkg_version", _raise)
    res = ts.detect_torch()
    # 原有 5 个 key 的语义锁死（下游 /api/torch/status + cli.py + 前端在消费）。
    # 不用 `==` 全等断言：ADR 0019 起 detect_torch 会附带后端字段（backend /
    # vendor_label / cuda_version / hip_version），那是加法，全等断言会把后续任何
    # 加字段都变成假失败。后端字段本身另有专门用例覆盖。
    assert res["installed"] is False
    assert res["version"] is None
    assert res["cuda_build"] is None
    assert res["cuda_available"] is False
    assert res["device_name"] is None


def test_detect_torch_cpu_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch 2.5.0+cpu → cuda_build='cpu', cuda_available=False。"""
    monkeypatch.setattr(ts, "_pkg_version", lambda _: "2.5.0+cpu")
    fake_torch = MagicMock()
    fake_torch.__version__ = "2.5.0+cpu"
    fake_torch.cuda.is_available.return_value = False
    fake_torch.version.cuda = None
    # 必须显式设 None：MagicMock 的属性访问会自动造一个 truthy 子 mock，
    # accelerator.detect() 读到 torch.version.hip 为真就判成 DCU（ADR 0019）。
    fake_torch.version.hip = None
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    res = ts.detect_torch()
    assert res["installed"] is True
    assert res["version"] == "2.5.0+cpu"
    assert res["cuda_build"] == "cpu"
    assert res["cuda_available"] is False


def test_detect_torch_cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch 2.5.0+cu128 + cuda 可用 → 全部 OK 字段。"""
    monkeypatch.setattr(ts, "_pkg_version", lambda _: "2.5.0+cu128")
    fake_torch = MagicMock()
    fake_torch.__version__ = "2.5.0+cu128"
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.get_device_name.return_value = "RTX 5090"
    fake_torch.version.cuda = "12.8"
    # 必须显式设 None：MagicMock 的属性访问会自动造一个 truthy 子 mock，
    # accelerator.detect() 读到 torch.version.hip 为真就判成 DCU（ADR 0019）。
    fake_torch.version.hip = None
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    res = ts.detect_torch()
    assert res["cuda_build"] == "cu128"
    assert res["cuda_available"] is True
    assert res["device_name"] == "RTX 5090"


def test_detect_torch_cuda_build_no_suffix_falls_back_to_version_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """老 torch __version__ 没 +suffix → 用 torch.version.cuda 推 cu tag。"""
    monkeypatch.setattr(ts, "_pkg_version", lambda _: "1.13.0")
    fake_torch = MagicMock()
    fake_torch.__version__ = "1.13.0"  # 没 + suffix
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.get_device_name.return_value = "Tesla T4"
    fake_torch.version.cuda = "11.8"
    # 必须显式设 None：MagicMock 的属性访问会自动造一个 truthy 子 mock，
    # accelerator.detect() 读到 torch.version.hip 为真就判成 DCU（ADR 0019）。
    fake_torch.version.hip = None
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    res = ts.detect_torch()
    assert res["cuda_build"] == "cu118"
    assert res["cuda_available"] is True


# ---------------------------------------------------------------------------
# current_status: 误装诊断
# ---------------------------------------------------------------------------


def test_current_status_flags_cpu_with_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU torch + 检测到 NVIDIA → is_cpu_with_gpu=True，UI 大警告。"""
    monkeypatch.setattr(ts, "detect_torch", lambda: {
        "installed": True, "version": "2.5.0+cpu", "cuda_build": "cpu",
        "cuda_available": False, "device_name": None,
    })
    monkeypatch.setattr(ts.onnxruntime_setup, "detect_cuda", lambda: {
        "available": True, "driver_version": "555.86", "gpu_name": "RTX 5090",
    })
    s = ts.current_status()
    assert s["is_cpu_with_gpu"] is True
    assert s["is_cuda_build_unavailable"] is False
    assert s["recommended_cu_tag"] == "cu128"


def test_current_status_flags_cuda_build_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """装 cu128 但 cuda.is_available()=False → is_cuda_build_unavailable=True。"""
    monkeypatch.setattr(ts, "detect_torch", lambda: {
        "installed": True, "version": "2.5.0+cu128", "cuda_build": "cu128",
        "cuda_available": False, "device_name": None,
    })
    monkeypatch.setattr(ts.onnxruntime_setup, "detect_cuda", lambda: {
        "available": True, "driver_version": "470.0", "gpu_name": "Tesla M40",
    })
    s = ts.current_status()
    assert s["is_cpu_with_gpu"] is False
    assert s["is_cuda_build_unavailable"] is True


def test_current_status_no_issue_when_cuda_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ts, "detect_torch", lambda: {
        "installed": True, "version": "2.5.0+cu128", "cuda_build": "cu128",
        "cuda_available": True, "device_name": "RTX 5090",
    })
    monkeypatch.setattr(ts.onnxruntime_setup, "detect_cuda", lambda: {
        "available": True, "driver_version": "555.86", "gpu_name": "RTX 5090",
    })
    s = ts.current_status()
    assert s["is_cpu_with_gpu"] is False
    assert s["is_cuda_build_unavailable"] is False
    assert s["cuda_available"] is True


# ---------------------------------------------------------------------------
# reinstall: 调 pip uninstall + install --index-url
# ---------------------------------------------------------------------------


def test_reinstall_invalid_target_raises() -> None:
    with pytest.raises(ValueError, match="非法 target"):
        ts.reinstall("xpu")


def test_reinstall_auto_picks_recommended(monkeypatch: pytest.MonkeyPatch) -> None:
    """target='auto' → 用 detect_cuda 驱动 → recommend_cu_tag。"""
    monkeypatch.setattr(ts.onnxruntime_setup, "detect_cuda", lambda: {
        "available": True, "driver_version": "555.86", "gpu_name": "RTX 5090",
    })
    pip_calls: list[list[str]] = []

    def fake_pip(args, **_kw):
        pip_calls.append(args)
        return 0, "ok"

    monkeypatch.setattr(ts, "_pip", fake_pip)
    monkeypatch.setattr(ts, "_pkg_version", lambda _: "2.5.0+cu128")

    res = ts.reinstall("auto")
    assert res["tag"] == "cu128"
    assert res["index_url"] == "https://download.pytorch.org/whl/cu128"
    assert res["restart_required"] is True
    # 第一 call uninstall，第二 call install
    assert pip_calls[0][:2] == ["uninstall", "-y"]
    assert "torch" in pip_calls[0] and "torchvision" in pip_calls[0]
    assert pip_calls[1][0] == "install"
    assert "--index-url" in pip_calls[1]
    assert "https://download.pytorch.org/whl/cu128" in pip_calls[1]


def test_reinstall_explicit_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    pip_calls: list[list[str]] = []
    monkeypatch.setattr(ts, "_pip", lambda args, **_kw: (pip_calls.append(args) or (0, "ok")))
    monkeypatch.setattr(ts, "_pkg_version", lambda _: "2.5.0+cu118")

    res = ts.reinstall("cu118")
    assert res["tag"] == "cu118"
    assert "https://download.pytorch.org/whl/cu118" in pip_calls[1]


def test_reinstall_cpu_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """cpu target 也走 PyTorch 自家 cpu index（避免 PyPI 默认歧义）。"""
    pip_calls: list[list[str]] = []
    monkeypatch.setattr(ts, "_pip", lambda args, **_kw: (pip_calls.append(args) or (0, "ok")))
    monkeypatch.setattr(ts, "_pkg_version", lambda _: "2.5.0+cpu")

    res = ts.reinstall("cpu")
    assert res["tag"] == "cpu"
    assert "https://download.pytorch.org/whl/cpu" in pip_calls[1]


def test_reinstall_pip_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pip(args, **_kw):
        if args[0] == "install":
            return 1, "ERROR: bad wheel"
        return 0, ""
    monkeypatch.setattr(ts, "_pip", fake_pip)
    with pytest.raises(RuntimeError, match="安装 torch"):
        ts.reinstall("cu128")


def test_cleanup_zombie_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """site-packages 里 `~*` 目录 + 文件应被清；正常包不动。"""
    fake_site = tmp_path / "site-packages"
    fake_site.mkdir()
    # 僵尸：~orch dir + ~orchvision dir + ~stray file
    (fake_site / "~orch-2.11.0.dist-info").mkdir()
    (fake_site / "~orch-2.11.0.dist-info" / "METADATA").write_text("fake")
    (fake_site / "~orchvision").mkdir()
    (fake_site / "~stray.txt").write_text("zombie file")
    # 正常包（不应动）
    (fake_site / "torch").mkdir()
    (fake_site / "torch" / "__init__.py").write_text("")
    (fake_site / "numpy").mkdir()

    monkeypatch.setattr(ts.sysconfig, "get_path", lambda _key: str(fake_site))
    cleaned = ts._cleanup_zombie_dirs()
    assert sorted(cleaned) == ["~orch-2.11.0.dist-info", "~orchvision", "~stray.txt"]
    assert not (fake_site / "~orch-2.11.0.dist-info").exists()
    assert not (fake_site / "~orchvision").exists()
    assert not (fake_site / "~stray.txt").exists()
    # 正常包没动
    assert (fake_site / "torch").exists()
    assert (fake_site / "numpy").exists()


def test_cleanup_zombie_dirs_empty_when_no_zombies(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fake_site = tmp_path / "site-packages"
    fake_site.mkdir()
    (fake_site / "torch").mkdir()
    monkeypatch.setattr(ts.sysconfig, "get_path", lambda _key: str(fake_site))
    assert ts._cleanup_zombie_dirs() == []


def test_reinstall_calls_cleanup_zombie(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """reinstall 应在 pip 前后都调一次 cleanup（双保险）。"""
    cleanup_calls: list[None] = []
    monkeypatch.setattr(
        ts, "_cleanup_zombie_dirs",
        lambda: (cleanup_calls.append(None), [])[1],
    )
    monkeypatch.setattr(ts, "_pip", lambda args, **_kw: (0, "ok"))
    monkeypatch.setattr(ts, "_pkg_version", lambda _: "2.5.0+cu128")

    res = ts.reinstall("cu128")
    # cleanup 调了两次（pip 前 + uninstall 后 install 前）
    assert len(cleanup_calls) == 2
    assert "cleaned_zombies" in res


def test_reinstall_returns_stdout_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    """stdout 长度 > 40 行时只保留尾部。"""
    long_log = "\n".join(f"line {i}" for i in range(100))
    monkeypatch.setattr(ts, "_pip", lambda args, **_kw: (0, long_log))
    monkeypatch.setattr(ts, "_pkg_version", lambda _: "2.5.0+cu128")

    res = ts.reinstall("cu128")
    tail_lines = res["stdout_tail"].splitlines()
    assert len(tail_lines) <= 40
    assert tail_lines[-1] == "line 99"


# ---------------------------------------------------------------------------
# DCU：把镜像预装的 HF 栈钉住（image_pinned_versions / write_dcu_constraints）
#
# 真机踩到的事（DTK 26.04）：requirements 的 `transformers>=4.57.0` 没有上限，
# 而 venv 看不见镜像预装的 4.57.6，于是 pip 拉了 5.16.1，连带把
# huggingface_hub 0.36.0→1.29.0（跨大版本）、tokenizers 0.22.2→0.23.1、
# safetensors 0.7.0→0.8.0。镜像里的 vllm 钉死 `transformers==4.57.6`，
# 于是那些 DCU 组件在运行时对不上。
#
# 与 torch/torchvision 的删行是两种机制：那两个装不了（PyPI 无 DTK wheel），
# 这几个装得上但版本要跟镜像一致 —— 所以用 pip 约束（-c），且约束能管住
# requirements.txt 里没有的传递依赖（tokenizers 就是）。
# ---------------------------------------------------------------------------


class _FakeDist:
    """importlib.metadata.Distribution 的最小替身（只要 metadata['Name'] + version）。"""

    def __init__(self, name: str, version: str):
        self.metadata = {"Name": name}
        self.version = version


def _fake_dcu(monkeypatch, *, in_venv: bool = True, base_exists: bool = True):
    """把 ts 伪装成「DCU + 在 venv 里 + base site-packages 存在」。"""
    monkeypatch.setattr(ts, "can_manage_torch_install", lambda: False)  # DCU
    if in_venv and base_exists:
        monkeypatch.setattr(ts, "_base_site_packages", lambda: __import__("pathlib").Path("/fake/base"))
    else:
        monkeypatch.setattr(ts, "_base_site_packages", lambda: None)


def _fake_dists(monkeypatch, dists: list[_FakeDist]):
    """替掉 importlib.metadata.distributions —— 它在函数体里 import，要打到源模块上。"""
    import importlib.metadata as im

    monkeypatch.setattr(im, "distributions", lambda path=None: iter(dists))


def test_image_pinned_versions_reads_base_site_packages(monkeypatch):
    """只报 _DCU_IMAGE_PINNED 里的包，版本取自 base site-packages。"""
    _fake_dcu(monkeypatch)
    _fake_dists(monkeypatch, [
        _FakeDist("transformers", "4.57.6"),
        _FakeDist("safetensors", "0.7.0"),
        _FakeDist("tokenizers", "0.22.2"),
        _FakeDist("huggingface_hub", "0.36.0"),   # 下划线要归一成横线
        _FakeDist("numpy", "1.26.4"),             # 不在名单里 → 不该出现
    ])
    got = ts.image_pinned_versions()
    assert got == {
        "transformers": "4.57.6",
        "safetensors": "0.7.0",
        "tokenizers": "0.22.2",
        "huggingface-hub": "0.36.0",
    }


def test_image_pinned_versions_empty_on_nvidia(monkeypatch):
    """非 DCU 一律不干预 —— NVIDIA 用户的安装行为逐字节不变。"""
    monkeypatch.setattr(ts, "can_manage_torch_install", lambda: True)

    def _boom():
        raise AssertionError("NVIDIA 路径不该去读 base site-packages")

    monkeypatch.setattr(ts, "_base_site_packages", _boom)
    assert ts.image_pinned_versions() == {}


def test_image_pinned_versions_empty_outside_venv(monkeypatch):
    """不在 venv 里就没有「镜像版 vs venv 版」之分，返回空。"""
    _fake_dcu(monkeypatch, in_venv=False)
    assert ts.image_pinned_versions() == {}


def test_image_pinned_versions_skips_packages_the_image_lacks(monkeypatch):
    """镜像没预装的包不出现在结果里 —— 不该约束，让 pip 自由装。

    这条是「约束」相对「删行」的关键优势：删行会让镜像没装该包的机器彻底装不上。
    """
    _fake_dcu(monkeypatch)
    _fake_dists(monkeypatch, [_FakeDist("transformers", "4.57.6")])
    assert ts.image_pinned_versions() == {"transformers": "4.57.6"}


def test_image_pinned_versions_survives_broken_metadata(monkeypatch):
    """元数据损坏时返回空而不是抛 —— 约束是加固，不该阻断安装。"""
    _fake_dcu(monkeypatch)
    import importlib.metadata as im

    def _boom(path=None):
        raise RuntimeError("metadata corrupt")

    monkeypatch.setattr(im, "distributions", _boom)
    assert ts.image_pinned_versions() == {}


def test_write_dcu_constraints_writes_pins(monkeypatch, tmp_path):
    """约束文件内容是 `name==version`，每包一行，且带说明注释。"""
    monkeypatch.setattr(ts, "image_pinned_versions", lambda: {
        "transformers": "4.57.6", "safetensors": "0.7.0",
    })
    dest = tmp_path / "sub" / "constraints.dcu.txt"
    out = ts.write_dcu_constraints(dest)
    assert out == dest and dest.is_file()          # 父目录要自动建
    text = dest.read_text(encoding="utf-8")
    assert "transformers==4.57.6" in text
    assert "safetensors==0.7.0" in text
    # 注释里要写明为什么，否则后人看到这个自动生成的文件会不敢动
    assert text.lstrip().startswith("#")
    assert "vllm" in text


def test_write_dcu_constraints_returns_none_when_nothing_to_pin(monkeypatch, tmp_path):
    """没什么要钉的时候**不建文件** —— 调用方据此决定是否传 -c。"""
    monkeypatch.setattr(ts, "image_pinned_versions", lambda: {})
    dest = tmp_path / "constraints.dcu.txt"
    assert ts.write_dcu_constraints(dest) is None
    assert not dest.exists()


def test_pinned_set_covers_the_packages_that_bit_us():
    """名单必须覆盖真机上被顶掉的那四个。

    这条是回归守卫：v0.25.0 合并后真机 pip 把这四个全顶了，其中 transformers
    4.x→5.x 与 huggingface_hub 0.x→1.x 都是破坏性大版本。
    """
    assert set(ts._DCU_IMAGE_PINNED) >= {
        "transformers", "safetensors", "tokenizers", "huggingface-hub",
    }
    # 名字必须是 pip 的规范形式（小写、横线）—— image_pinned_versions 按此比对
    for name in ts._DCU_IMAGE_PINNED:
        assert name == name.lower().replace("_", "-"), f"{name} 不是规范包名"
