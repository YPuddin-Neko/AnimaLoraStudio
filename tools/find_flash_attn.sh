#!/usr/bin/env bash
# 在海光 DCU（DTK）环境里找 flash-attn：已装？磁盘上有包/库？能 import？
#
# 为什么需要这个脚本：DTK 的 torch 编译时**开启**了 SDPA 的 flash 后端，但把 kernel
# 委托给外部 `flash_attn_2_cuda*.so`（海光把 flash-attn 作为独立包发布，不在 PyPI）。
# 包没装时的表现很有欺骗性 —— `torch.backends.cuda.flash_sdp_enabled()` 仍报 True，
# 而真正调用 SDPA 会抛：
#     RuntimeError: No matching libraries found for flash_attn_2_cuda*.so
# 本项目已用 utils/accelerator.py:configure_sdpa() 兜住这种情况（启动期实测各后端，
# 把不可用的关掉，退到 math 后端）。所以 flash-attn **不是训练的前置条件**，装上是提速：
#   1. 恢复 SDPA 的 flash 后端（比 math 快、显存占用低得多）
#   2. 让项目的 attention_backend=flash_attn 路径可用（比 SDPA 更快）
#
# 用法：bash tools/find_flash_attn.sh
#
# 只读脚本：不装包、不改环境、不写文件。
#
# NOTE: echo messages kept in plain ASCII/English, matching studio.sh's convention
#       (non-UTF-8 locales would otherwise render them as garbled bytes).

set -u

# 只搜这几个目录，不做 `find /` 全盘扫描：答案只可能在这里，而全盘扫描在容器里
# 又慢又会撞一堆权限错。
#
# 刻意**不用** `find -xdev`：容器里 /opt/dtk 与 site-packages 常是独立挂载点，
# -xdev 不跨文件系统，正好会把最该搜的目录跳过。
_SEARCH_DIRS=(
    /opt                                  # DTK 本体与厂商包常驻处
    /root                                 # 手动下载的 wheel 习惯放这
    "$HOME/.cache/pip"                    # pip 缓存（装过就有，即便后来卸了）
)

# site-packages 由当前解释器自报，避免写死 python3.11 之类的版本号
_SITE="$(python -c 'import sysconfig;print(sysconfig.get_path("purelib"))' 2>/dev/null || true)"
[ -n "$_SITE" ] && _SEARCH_DIRS+=("$_SITE")

_found_so=""
_found_pkg=""

echo "=============================================================="
echo " 1. flash_attn 的 .so   <- torch dlopen 的正是这个文件"
echo "=============================================================="
for d in "${_SEARCH_DIRS[@]}"; do
    [ -e "$d" ] || continue
    # 每个 find 是独立进程，`head` 触发的 SIGPIPE 只终止当前这一个，
    # 不会像 `{ ...; } | head` 那样把整段脚本掐断。
    _hits="$(find "$d" -name 'flash_attn*.so*' 2>/dev/null | head -20)"
    if [ -n "$_hits" ]; then
        echo "$_hits"
        _found_so="yes"
    fi
done
[ -z "$_found_so" ] && echo "(none)"

echo
echo "=============================================================="
echo " 2. wheel / source package on disk"
echo "=============================================================="
for d in "${_SEARCH_DIRS[@]}"; do
    [ -e "$d" ] || continue
    _hits="$(find "$d" \( -iname '*flash*attn*.whl' -o -iname '*flash*attn*.tar.gz' \) \
             2>/dev/null | head -20)"
    if [ -n "$_hits" ]; then
        echo "$_hits"
        _found_pkg="yes"
    fi
done
[ -z "$_found_pkg" ] && echo "(none)"

echo
echo "=============================================================="
echo " 3. DTK vendor wheel directories"
echo "=============================================================="
# DTK 镜像有时把配套 wheel（flash-attn / deepspeed / apex ...）预放在这些目录，
# 装的时候不需要联网。列出来顺便看看里面有什么。
_wheel_dirs="$(ls -d /opt/dtk*/wheel* /opt/wheels /root/wheels /opt/hyhal/wheel* 2>/dev/null)"
if [ -n "$_wheel_dirs" ]; then
    echo "$_wheel_dirs"
    echo "--- contents ---"
    # shellcheck disable=SC2086  故意按空白拆成多个参数
    ls -1 $_wheel_dirs 2>/dev/null | head -40
else
    echo "(none)"
fi

echo
echo "=============================================================="
echo " 4. installed packages (attention / triton / DTK-flavoured)"
echo "=============================================================="
# grep 关键词覆盖：flash-attn 各种命名、triton（SDPA 部分后端与 torch.compile 用）、
# das/dtk（海光给自家 build 打的本地版本标签，如 2.9.0+das.dtk2604）
pip list 2>/dev/null | grep -iE 'flash|attn|triton|das|dtk' || echo "(none matched)"

echo
echo "=============================================================="
echo " 5. import check"
echo "=============================================================="
python - <<'PY'
try:
    import flash_attn, os
    print("import flash_attn: OK")
    print("  version:", getattr(flash_attn, "__version__", "?"))
    print("  path   :", os.path.dirname(flash_attn.__file__))
except Exception as exc:
    print(f"import flash_attn: FAILED ({type(exc).__name__}: {exc})")

# 这一段才是真正决定 SDPA flash 后端能不能用的检查 —— 直接问 torch，
# 不靠「包装没装」间接推断（DTK 上两者会不一致，正是本脚本存在的原因）。
try:
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    print()
    print("torch:", torch.__version__, "| hip:", getattr(torch.version, "hip", None))
    if not torch.cuda.is_available():
        print("torch.cuda.is_available() = False -> cannot probe SDPA backends")
    else:
        q = torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.bfloat16)
        for name, backend in (
            ("flash        ", SDPBackend.FLASH_ATTENTION),
            ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
            ("math         ", SDPBackend.MATH),
        ):
            try:
                with sdpa_kernel(backend):
                    F.scaled_dot_product_attention(q, q, q)
                print(f"  SDPA {name}: OK")
            except Exception as exc:
                print(f"  SDPA {name}: FAILED ({type(exc).__name__}: {str(exc)[:90]})")
except Exception as exc:
    print(f"SDPA probe skipped ({type(exc).__name__}: {exc})")
PY

echo
echo "=============================================================="
echo " What to do next"
echo "=============================================================="
if [ -n "$_found_so" ]; then
    echo "* A flash_attn .so exists on disk (section 1)."
    echo "  If 'import flash_attn' failed anyway, the package dir is likely not on"
    echo "  sys.path -- check which venv/interpreter owns that path."
elif [ -n "$_found_pkg" ]; then
    echo "* A wheel/tarball exists on disk (section 2). Install it directly:"
    echo "      pip install <path-from-section-2>"
    echo "  Then re-run this script; SDPA flash should flip to OK."
else
    echo "* Nothing found locally. Get the flash-attn build matching this image's DTK"
    echo "  version from the Hygon developer channel (SourceFind / DTK companion repo)."
fi
echo
echo "* Not urgent: training works without it. AnimaLoraStudio probes SDPA backends at"
echo "  startup (utils/accelerator.py:configure_sdpa) and falls back to the math"
echo "  backend, which is slower and uses more VRAM on long sequences but is correct."
echo "  Installing flash-attn is purely a speed/VRAM win."
