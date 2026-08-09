#!/usr/bin/env bash
# AnimaStudio Linux/macOS shortcut -- forwards to: python -m studio
# Usage:
#   ./studio.sh [--mirror] [--reinstall] [subcommand]
#
#   --mirror          Use Tencent pip mirror during first-run setup.
#                     Without this flag, official PyPI is tried first; the mirror is
#                     used as a fallback if the official source fails.
#
#   --reinstall       DELETE venv/ and rebuild from scratch (studio_data/ kept).
#                     Use when venv is broken beyond repair (dep conflict / corrupt
#                     wheels / etc). Asks for confirmation.
#
#   --torch=<tag>     Force a specific PyTorch CUDA wheel on first-run venv setup
#                     AND (via Python cli) reinstall if the current torch differs.
#                     Tags: cu128 cu126 cu124 cu118 cpu
#                     Use this on CPU-only rentals when you want GPU torch pre-installed
#                     for a later GPU machine.  Example: ./studio.sh --torch=cu128
#                     REFUSED on Hygon DCU boxes (DTK torch ships inside the vendor
#                     image and cannot be reinstalled with pip).
#
#   subcommand: run (default) | dev | build | test
#
#   run subcommand flags:
#     --port <N>      backend uvicorn port (default 8765)
#     --host <H>      bind host (default 127.0.0.1)
#     --no-browser    do not auto-open browser
#     --no-build      skip frontend rebuild check
#     --torch <tag>   force torch CUDA tag (cu128/cu126/cu124/cu118/cpu)
#
#   dev subcommand flags:
#     --port <N>      backend uvicorn port (default 8765)
#     --fe-port <N>   frontend Vite dev server port (default 5173)
#     --host <H>      bind host (default 127.0.0.1)
#     --no-browser    do not auto-open browser
#     --torch <tag>   force torch CUDA tag (cu128/cu126/cu124/cu118/cpu)
#
# Safe to run with either ./studio.sh or `bash studio.sh`.
# Avoid `source studio.sh` -- not needed (we call venv python directly).
#
# NOTE: shell echo messages are kept in plain ASCII/English so non-UTF-8
#       locales don't render them as garbled bytes. Python-side messages are
#       UTF-8 (PYTHONUTF8=1 / PYTHONIOENCODING=utf-8 below).
#
# HYGON DCU (DTK) NOTE -- the single most destructive failure mode of this
# script on a DCU box, and how it is prevented:
#
#   The DTK build of torch/torchvision ships INSIDE the vendor container image
#   (e.g. pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.11). Those wheels do not exist
#   on PyPI and are tied to the DTK runtime in the image. A plain
#   `pip install -r requirements.txt` sees `torch>=2.0.0`, pulls the PyPI CPU
#   wheel, and overwrites them -- unrecoverable via pip, the user has to rebuild
#   the container.
#
#   Two things go wrong independently, so we fix both:
#     1. `python -m venv venv` hides system site-packages by default, so the
#        pre-installed DTK torch is invisible inside venv/ and pip happily
#        "installs the missing torch". -> create the venv with
#        --system-site-packages on DCU.
#     2. Even with torch visible, a transitive dep resolution could still try to
#        touch torch/torchvision. -> feed pip a filtered requirements copy with
#        those two lines removed (_prepare_requirements below).
#
#   Backend detection is asked of tools/select_torch_index.py --backend, which
#   forwards to utils/accelerator.py (single source of truth for the repo).
#   NVIDIA / CPU paths are unchanged: same commands, same order, same output.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "studio.sh: cannot cd to $SCRIPT_DIR" >&2; exit 1; }

# Mirror studio.bat's `pause` on error: when launched from a file manager
# (double-click) the terminal closes on exit and the user can't read the
# error (e.g. missing Node.js / Python). EXIT trap covers every `exit N`
# path (setup-time `exit 1`, main-loop non-zero rc) uniformly; only pause
# when stdin is a TTY so CI / piped invocations don't hang.
_pause_if_tty_on_error() {
    local rc=$?
    # Drop the filtered requirements temp file (DCU only; see
    # _prepare_requirements). Done in the trap rather than inline so every exit
    # path is covered, including the setup-time `exit 1`s.
    if [ -n "$_REQ_FILTERED" ]; then
        rm -f "$_REQ_FILTERED"
    fi
    # 130 = SIGINT (Ctrl+C), 143 = SIGTERM — user wanted to kill, don't make
    # them press Enter to dismiss.
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 130 ] || [ "$rc" -eq 143 ]; then
        return
    fi
    if [ -t 0 ] && [ -t 1 ]; then
        printf "[studio] Press Enter to close..." >&2
        read -r _ || true
    fi
}
trap _pause_if_tty_on_error EXIT

# Force Python UTF-8 output so cli.py messages with non-ASCII characters are
# not mangled on non-UTF-8 locales.
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# Parse our flags; collect remaining args to forward to Python.
_USE_MIRROR=0
_REINSTALL=0
_TORCH_TAG=""
_PASSTHROUGH=()
for _arg in "$@"; do
    case "$_arg" in
        --mirror)    _USE_MIRROR=1 ;;
        --reinstall) _REINSTALL=1 ;;
        --torch=*)   _TORCH_TAG="${_arg#--torch=}"
                     _PASSTHROUGH+=("--torch" "$_TORCH_TAG") ;;
        *)           _PASSTHROUGH+=("$_arg") ;;
    esac
done

_TENCENT="https://mirrors.cloud.tencent.com/pypi/simple/"
_REQ_MARKER="venv/.studio-requirements.sha256"

# Accelerator backend: cuda | dcu | cpu. Empty until _detect_backend runs.
_BACKEND=""
# Path of the filtered requirements temp file, if one was generated (DCU only).
_REQ_FILTERED=""

_detect_backend() {
    # Usage: _detect_backend <python>
    # Memoized: probing runs nvidia-smi / hy-smi, so only pay for it on the paths
    # that actually install packages (fresh venv, stale dep sync). The plain
    # launch path must stay as fast as before.
    [ -n "$_BACKEND" ] && return 0
    _BACKEND="$("$1" tools/select_torch_index.py --backend 2>/dev/null || true)"
    # Empty means the helper itself failed (missing file / broken interpreter).
    # Assume cuda: that is the pre-DCU behaviour of this script, so a broken
    # probe degrades to "exactly what we did before" instead of something new.
    [ -z "$_BACKEND" ] && _BACKEND="cuda"
    [ "$_BACKEND" = "dcu" ] && echo "[studio] setup: Hygon DCU (DTK) detected; PyTorch is managed by the container image, not by pip"
    return 0
}

_prepare_requirements() {
    # Sets _REQ_ARG to the requirements file pip should be given.
    # cuda / cpu: requirements.txt unchanged.
    # dcu: a filtered copy under tmp/ with the torch + torchvision lines dropped,
    #      because those two are pre-installed DTK wheels that pip must never
    #      touch (see the HYGON DCU NOTE at the top of this file).
    # requirements.txt itself is NEVER rewritten -- it stays the correct
    # constraint set for NVIDIA users, and check_requirements_changed.py hashes
    # the original file (torch being unmanaged on DCU is intentional, so the
    # marker should keep tracking upstream edits to the real file).
    # Sets a global instead of echoing, so the temp path is visible to the EXIT
    # trap that removes it (a $(...) subshell assignment would be lost).
    _REQ_ARG="requirements.txt"
    [ "$_BACKEND" != "dcu" ] && return 0
    if [ -z "$_REQ_FILTERED" ]; then
        mkdir -p tmp
        _REQ_FILTERED="tmp/requirements.dcu.txt"
        # Anchor on the line start and require a version-spec / EOL right after
        # the name: a bare `^torch` prefix match would also eat `torchsde`,
        # which is a pure-Python package that we DO need from PyPI.
        grep -Ev '^[[:space:]]*(torch|torchvision)[[:space:]]*([<>=!~;[].*)?$' \
            requirements.txt > "$_REQ_FILTERED" || true
        echo "[studio] setup: torch/torchvision removed from the pip requirement list (image-provided DTK wheels)"
    fi
    _REQ_ARG="$_REQ_FILTERED"
    return 0
}

_pip_install() {
    # Usage: _pip_install [pip args...]
    # Tries official PyPI first; falls back to Tencent mirror on failure.
    # With --mirror: goes straight to Tencent mirror.
    if [ "$_USE_MIRROR" = "1" ]; then
        echo "[studio] setup: using Tencent mirror for pip"
        "$PYTHON" -m pip install "$@" -i "$_TENCENT"
    else
        "$PYTHON" -m pip install "$@" || {
            echo "[studio] setup: pip failed, retrying via Tencent mirror..."
            "$PYTHON" -m pip install "$@" -i "$_TENCENT"
        }
    fi
}

# --reinstall: nuke venv before detection. studio_data/ is untouched.
if [ "$_REINSTALL" = "1" ] && [ -d venv ]; then
    echo "[studio] --reinstall: venv/ will be DELETED and rebuilt."
    echo "[studio]   - studio_data/ (your projects + LoRA weights) is NOT touched"
    echo "[studio]   - any user-installed pip packages outside requirements.txt will be lost"
    printf "Continue? [y/N] "
    read -r _ans
    case "$_ans" in
        [yY]*) ;;
        *)     echo "[studio] --reinstall aborted"; exit 0 ;;
    esac
    echo "[studio] removing venv/..."
    rm -rf venv || { echo "studio.sh: failed to remove venv" >&2; exit 1; }
fi

_check_venv_python_version() {
    # Warn if existing venv's Python is < 3.10. User can fix with --reinstall.
    if ! "$PYTHON" -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >/dev/null 2>&1; then
        echo "[studio] WARNING: venv/ uses Python < 3.10; some deps may fail to install" >&2
        echo "[studio] consider ./studio.sh --reinstall to recreate with a newer Python" >&2
    fi
}

if [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
    _check_venv_python_version
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    _check_venv_python_version
else
    # PR-S0: iterate explicit versions first so users with multiple Python
    # installs (Ubuntu pre-22.04 has python3=3.8 even when python3.10 is also
    # installed) get the latest >= 3.10. Fall back to python3 / python.
    BOOTSTRAP_PY=""
    for _candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
        if command -v "$_candidate" >/dev/null 2>&1; then
            if "$_candidate" -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >/dev/null 2>&1; then
                BOOTSTRAP_PY="$_candidate"
                break
            fi
        fi
    done
    if [ -z "$BOOTSTRAP_PY" ]; then
        echo "studio.sh: no Python 3.10+ found on PATH (need one of python3.10/3.11/3.12/3.13)" >&2
        exit 1
    fi
    # Probe with the bootstrap interpreter: the venv does not exist yet, and the
    # helper is stdlib-only so any Python can run it.
    _detect_backend "$BOOTSTRAP_PY"

    if [ "$_BACKEND" = "dcu" ] && [ -n "$_TORCH_TAG" ]; then
        echo "studio.sh: --torch=$_TORCH_TAG is not supported on Hygon DCU." >&2
        echo "  The DTK build of torch is pre-installed in the container image and is not" >&2
        echo "  published on PyPI. Installing a PyPI wheel would replace it with a CPU or" >&2
        echo "  NVIDIA build and break the environment beyond pip repair (container rebuild" >&2
        echo "  required). To change torch version, use a container image with the DTK" >&2
        echo "  version you want." >&2
        exit 1
    fi

    _VENV_ARGS=()
    if [ "$_BACKEND" = "dcu" ]; then
        # Without this the pre-installed DTK torch is invisible inside venv/ and
        # pip would "helpfully" install the PyPI CPU wheel over it. This is the
        # main reason DCU needs a different venv creation call at all.
        _VENV_ARGS+=(--system-site-packages)
        echo "[studio] setup: creating venv with --system-site-packages so the image's DTK torch stays visible"
    fi
    echo "[studio] No venv found. Creating venv/ via $BOOTSTRAP_PY ..."
    "$BOOTSTRAP_PY" -m venv "${_VENV_ARGS[@]}" venv || { echo "studio.sh: failed to create venv" >&2; exit 1; }
    PYTHON="venv/bin/python"

    _pip_install --upgrade pip || { echo "studio.sh: failed to upgrade pip" >&2; exit 1; }

    # GPU-aware torch first install (PR-S1a). Without this, requirements.txt's
    # bare `torch>=2.0.0` makes pip pull the CPU wheel from PyPI default. By
    # installing torch from PyTorch's CUDA index FIRST, the requirements.txt
    # constraint is already satisfied and pip won't replace it.
    # --torch=<tag> overrides auto-detection (useful on CPU-only rentals).
    #
    # On DCU this whole step is skipped: select_torch_index.py prints nothing
    # there by design, and --torch was already rejected above.
    if [ "$_BACKEND" = "dcu" ]; then
        echo "[studio] setup: skipping torch install (DTK torch comes from the container image)"
    elif [ -n "$_TORCH_TAG" ]; then
        _TORCH_INDEX="https://download.pytorch.org/whl/$_TORCH_TAG"
        echo "[studio] setup: --torch=$_TORCH_TAG specified; installing torch from $_TORCH_INDEX"
        if ! _pip_install torch torchvision --index-url "$_TORCH_INDEX"; then
            echo "[studio] setup: forced torch install failed; will fall back to PyPI default in requirements.txt"
        fi
    else
        _TORCH_INDEX="$("$PYTHON" tools/select_torch_index.py 2>/dev/null || true)"
        if [ -n "$_TORCH_INDEX" ]; then
            echo "[studio] setup: NVIDIA GPU detected; installing torch from $_TORCH_INDEX"
            if ! "$PYTHON" -m pip install torch torchvision --index-url "$_TORCH_INDEX"; then
                echo "[studio] setup: CUDA torch install failed; will fall back to PyPI default in requirements.txt"
                echo "[studio] setup: you can fix manually later via Studio Settings > PyTorch > Reinstall"
            fi
        fi
    fi

    if [ -f requirements.txt ]; then
        echo "[studio] Installing Python dependencies..."
        _prepare_requirements
        _pip_install -r "$_REQ_ARG" || { echo "studio.sh: pip install failed" >&2; exit 1; }
    else
        echo "studio.sh: requirements.txt not found, skipping dependency install" >&2
    fi
    # PR-S1b: write hash marker after fresh install so future stale check is correct
    "$PYTHON" tools/check_requirements_changed.py --marker "$_REQ_MARKER" --update-marker >/dev/null 2>&1 || true
fi

# PR-S1b: stale check. If requirements.txt content hash differs from the marker
# (or no marker yet on an old venv), `pip install -r requirements.txt` to add
# missing packages. NO --upgrade -- existing torch+cu128 etc stays untouched.
_STALE="$("$PYTHON" tools/check_requirements_changed.py --marker "$_REQ_MARKER" 2>/dev/null || echo missing)"
if [ "$_STALE" = "stale" ]; then
    echo "[studio] requirements.txt changed since last sync; installing new deps (no upgrade)..."
    # Backend probe happens here (not at script start) so the normal launch path
    # pays no smi call. On DCU this is what keeps a dep sync from overwriting the
    # image's DTK torch -- the marker is also stale on every OLD venv that
    # predates this check, so this path is hit on real DCU upgrades too.
    _detect_backend "$PYTHON"
    _prepare_requirements
    if _pip_install -r "$_REQ_ARG"; then
        "$PYTHON" tools/check_requirements_changed.py --marker "$_REQ_MARKER" --update-marker >/dev/null 2>&1 || true
        echo "[studio] dep sync complete"
    else
        echo "[studio] WARNING: dep sync failed; existing venv still works but may miss new deps" >&2
        echo "[studio] try ./studio.sh --reinstall if errors persist" >&2
    fi
fi

echo "studio.sh: using $PYTHON"

# Restart loop (PR-A): if cli.py exits but tmp/restart is still present, loop
# back and re-run. cli.py's own inner loop handles the common case (server
# requests restart from /api/system/restart); this outer loop is the safety net.
#
# Special exit code 42 (PR-D, installer self-update): cli.py detected that
# cli.py / studio.sh / studio.bat itself was just replaced by `git reset`, and
# kept tmp/restart so we'd see it. We `exec` ourselves so the new wrapper code
# is loaded from disk (bash has the old loop body in memory; the new wrapper
# might have different bootstrap / dep-install logic). See ADR 0002.
# We exec with _PASSTHROUGH only (not original "$@") so --reinstall does not
# get re-triggered.
while true; do
    "$PYTHON" -m studio "${_PASSTHROUGH[@]}"
    EXIT_CODE=$?
    if [ ! -f tmp/restart ]; then
        break
    fi
    if [ "$EXIT_CODE" -eq 42 ]; then
        echo "[studio] launcher updated, re-exec wrapper"
        rm -f tmp/restart
        exec "$0" "${_PASSTHROUGH[@]}"
    fi
    echo "[studio] restart requested (wrapper loop)"
    rm -f tmp/restart
done

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "[studio] Exit code $EXIT_CODE, see error messages above."
fi
exit $EXIT_CODE
