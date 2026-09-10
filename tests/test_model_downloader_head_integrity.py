"""Head detector integrity caching uses temporary files, never real weights/network."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from studio.services.models import downloader as dl


@pytest.fixture
def integrity_env(tmp_path, monkeypatch):
    payload = b"verified-model"
    monkeypatch.setattr(dl, "HEAD_DETECTOR_SIZE", len(payload))
    monkeypatch.setattr(dl, "HEAD_DETECTOR_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(dl, "_HEAD_INTEGRITY_CACHE", dl.OrderedDict())
    target = dl.head_detector_target(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    hashes = []
    real_sha256 = hashlib.sha256

    def counted_hash():
        hashes.append(True)
        return real_sha256()

    monkeypatch.setattr(dl.hashlib, "sha256", counted_hash)
    return tmp_path, target, payload, hashes


def test_unchanged_status_reuses_hash_and_returns_independent_dict(integrity_env):
    root, _target, _payload, hashes = integrity_env
    first = dl.head_detector_status(root)
    assert first["valid"]
    first["valid"] = False
    for _ in range(5):
        assert dl.head_detector_status(root)["valid"]
    assert len(hashes) == 1
    assert dl.head_detector_status(root, force_verify=True)["valid"]
    assert len(hashes) == 2


def test_same_size_replacement_and_deletion_invalidate_cache(integrity_env):
    root, target, payload, hashes = integrity_env
    assert dl.head_detector_status(root)["valid"]
    old_stat = target.stat()
    replacement = target.with_suffix(".replacement")
    replacement.write_bytes(b"x" * len(payload))
    os.utime(replacement, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    os.replace(replacement, target)
    assert not dl.head_detector_status(root)["valid"]
    assert len(hashes) == 2
    target.unlink()
    assert dl.head_detector_status(root) == {
        "exists": False, "valid": False, "size": 0, "mtime": 0.0,
    }
    assert not dl._HEAD_INTEGRITY_CACHE
    target.write_bytes(payload)
    assert dl.head_detector_status(root)["valid"]
    assert len(hashes) == 3


def test_metadata_and_expected_pins_invalidate_cache(integrity_env, monkeypatch):
    root, target, _payload, hashes = integrity_env
    assert dl.head_detector_status(root)["valid"]
    st = target.stat()
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert dl.head_detector_status(root)["valid"]
    monkeypatch.setattr(dl, "HEAD_DETECTOR_REVISION", "new-revision")
    assert dl.head_detector_status(root)["valid"]
    monkeypatch.setattr(dl, "HEAD_DETECTOR_SHA256", "0" * 64)
    assert not dl.head_detector_status(root)["valid"]
    assert len(hashes) == 4
    monkeypatch.setattr(dl, "HEAD_DETECTOR_SIZE", 1)
    assert not dl.head_detector_status(root)["valid"]
    assert not dl._HEAD_INTEGRITY_CACHE


def test_cache_is_bounded_lru(integrity_env, monkeypatch):
    root, _target, payload, hashes = integrity_env
    monkeypatch.setattr(dl, "_HEAD_INTEGRITY_CACHE_LIMIT", 2)
    roots = [root / str(idx) for idx in range(3)]
    for item in roots:
        path = dl.head_detector_target(item)
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
    for item in roots[:2]:
        assert dl.head_detector_status(item)["valid"]
    assert dl.head_detector_status(roots[0])["valid"]
    assert dl.head_detector_status(roots[2])["valid"]
    assert len(dl._HEAD_INTEGRITY_CACHE) == 2
    assert dl.head_detector_status(roots[0])["valid"]
    assert len(hashes) == 3
    assert dl.head_detector_status(roots[1])["valid"]
    assert len(hashes) == 4


def test_transient_read_error_is_not_cached(integrity_env, monkeypatch):
    root, target, _payload, hashes = integrity_env
    real_open = Path.open

    def unreadable(path, *args, **kwargs):
        if path == target:
            raise PermissionError("temporarily locked")
        return real_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", unreadable)
        assert not dl.head_detector_status(root)["valid"]
    assert not dl._HEAD_INTEGRITY_CACHE
    assert dl.head_detector_status(root)["valid"]
    assert len(hashes) == 1


def test_file_mutation_during_hash_is_not_accepted_or_cached(integrity_env, monkeypatch):
    root, target, _payload, _hashes = integrity_env
    real_sha256 = dl.hashlib.sha256

    class MutatingHash:
        def __init__(self):
            self.digest = real_sha256()

        def update(self, chunk):
            self.digest.update(chunk)
            st = target.stat()
            os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

        def hexdigest(self):
            return self.digest.hexdigest()

    with monkeypatch.context() as patch:
        patch.setattr(dl.hashlib, "sha256", MutatingHash)
        assert not dl.head_detector_status(root)["valid"]
    assert not dl._HEAD_INTEGRITY_CACHE
    assert dl.head_detector_status(root)["valid"]


def test_path_and_descriptor_ctime_clocks_may_differ(integrity_env, monkeypatch):
    from types import SimpleNamespace

    root, _target, _payload, hashes = integrity_env
    real_fstat = os.fstat

    def different_ctime(fd):
        st = real_fstat(fd)
        return SimpleNamespace(
            st_dev=st.st_dev, st_ino=st.st_ino, st_mode=st.st_mode,
            st_size=st.st_size, st_mtime_ns=st.st_mtime_ns,
            st_ctime_ns=st.st_ctime_ns + 1_000_000,
        )

    monkeypatch.setattr(dl.os, "fstat", different_ctime)
    assert dl.head_detector_status(root)["valid"]
    assert dl.head_detector_status(root)["valid"]
    assert len(hashes) == 1


def test_download_completion_forces_hash_even_if_catalog_cached_it(integrity_env, monkeypatch):
    root, target, payload, hashes = integrity_env
    target.unlink()

    def download(_repo, _subpath, path, **kwargs):
        assert kwargs["revision"] == dl.HEAD_DETECTOR_REVISION
        path.write_bytes(payload)
        # Simulate a catalog request between download and final verification.
        assert dl.head_detector_status(root)["valid"]
        return True

    monkeypatch.setattr(dl._sources, "download_flat", download)
    assert dl.download_head_detector(root)
    assert len(hashes) == 2
    assert dl.head_detector_status(root)["valid"]
    assert len(hashes) == 2


def test_custom_detector_cannot_overwrite_or_delete_builtin(tmp_path, monkeypatch):
    builtin = dl.head_detector_target(tmp_path)
    builtin.parent.mkdir(parents=True)
    builtin.write_bytes(b"pinned")
    downloads: list[Path] = []

    def download(_repo, _filename, target, **_kwargs):
        downloads.append(target)
        target.write_bytes(b"custom")
        return True

    monkeypatch.setattr(dl._sources, "download_flat", download)
    monkeypatch.setattr(dl, "models_root", lambda: tmp_path)

    assert not dl.download_head_detector_custom(
        "hf", "owner/detector", "model.onnx", tmp_path,
    )
    assert downloads == []
    assert builtin.read_bytes() == b"pinned"
    with pytest.raises(ValueError, match="reserved"):
        dl.delete_asset("head_detector_custom", "model.onnx")
    assert builtin.read_bytes() == b"pinned"

    assert dl.download_head_detector_custom(
        "hf", "owner/detector", "custom.onnx", tmp_path,
    )
    assert downloads == [builtin.with_name("custom.onnx")]
    assert downloads[0].read_bytes() == b"custom"
    assert builtin.read_bytes() == b"pinned"
