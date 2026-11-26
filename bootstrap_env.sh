#!/usr/bin/env bash
# bootstrap_env.sh — Offline environment check, skill link verification & smoke test
# Safe to run on any Ubuntu 22.04 / 24.04 machine (zero ROS 2 or SocketCAN I/O).
set -euo pipefail

WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "$WORKSPACE_ROOT/.venv/bin/python" ]] && \
       "$WORKSPACE_ROOT/.venv/bin/python" -c "import numpy, scipy, cv2, pytest, yaml" >/dev/null 2>&1; then
        PYTHON_BIN="$WORKSPACE_ROOT/.venv/bin/python"
    elif [[ -x "$WORKSPACE_ROOT/.venv-pika-dataset/bin/python" ]] && \
       "$WORKSPACE_ROOT/.venv-pika-dataset/bin/python" -c "import numpy, scipy, cv2, pytest, yaml" >/dev/null 2>&1; then
        PYTHON_BIN="$WORKSPACE_ROOT/.venv-pika-dataset/bin/python"
    elif [[ -x "$WORKSPACE_ROOT/.venv-pika/bin/python" ]] && \
       "$WORKSPACE_ROOT/.venv-pika/bin/python" -c "import numpy, scipy, cv2, pytest, yaml" >/dev/null 2>&1; then
        PYTHON_BIN="$WORKSPACE_ROOT/.venv-pika/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        echo "[ERROR] python3 not found. Install Python 3 first:" >&2
        echo "  sudo apt-get update && sudo apt-get install -y python3 python3-pip" >&2
        exit 1
    fi
fi

echo "=================================================================="
echo " quang_ws Handover Bootstrap & Health Check"
echo " Workspace : $WORKSPACE_ROOT"
echo " Python    : $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"
echo "=================================================================="

# ------------------------------------------------------------------------------
# 1. Check Python & System Dependencies
# ------------------------------------------------------------------------------
echo ""
echo "[1/5] Checking Python and system dependencies..."

missing_core=()
missing_optional=()

check_py_mod() {
    local mod="$1"
    local tier="$2"
    if "$PYTHON_BIN" -c "import $mod" >/dev/null 2>&1; then
        printf "  [OK]      python module '%s'\n" "$mod"
    else
        if [[ "$tier" == "core" ]]; then
            printf "  [MISSING] python module '%s' (REQUIRED for bridge & offline tests)\n" "$mod"
            missing_core+=("$mod")
        else
            printf "  [WARN]    python module '%s' (needed for HDF5/LeRobot dataset export)\n" "$mod"
            missing_optional+=("$mod")
        fi
    fi
}

check_py_mod numpy core
check_py_mod scipy core
check_py_mod cv2 core
check_py_mod pytest core
check_py_mod yaml core
check_py_mod h5py optional

for cmd in git docker uv; do
    if command -v "$cmd" >/dev/null 2>&1 || [[ "$cmd" == "uv" && -x "$HOME/.local/bin/uv" ]]; then
        printf "  [OK]      system command '%s'\n" "$cmd"
    else
        printf "  [INFO]    optional tool '%s' not found on PATH\n" "$cmd"
    fi
done

if [[ -f /opt/ros/humble/setup.bash ]]; then
    printf "  [OK]      ROS 2 Humble detected (/opt/ros/humble)\n"
elif [[ -f /opt/ros/jazzy/setup.bash ]]; then
    printf "  [INFO]    ROS 2 Jazzy detected (/opt/ros/jazzy); note piper_ros-humble targets ROS 2 Humble\n"
else
    printf "  [INFO]    ROS 2 not detected under /opt/ros (only required for live Dual-Piper CAN control)\n"
fi

if [[ "${#missing_core[@]}" -gt 0 ]]; then
    echo ""
    echo "[ERROR] Missing required Python packages: ${missing_core[*]}" >&2
    echo "Install them on Ubuntu 24.04 / 22.04 with:" >&2
    echo "  sudo apt-get update && sudo apt-get install -y \\" >&2
    echo "    python3-numpy python3-scipy python3-opencv python3-pytest python3-yaml python3-h5py python3-tk" >&2
    exit 1
fi

if [[ "${#missing_optional[@]}" -gt 0 ]]; then
    echo ""
    echo "  -> Note: Optional export module(s) missing (${missing_optional[*]})."
    echo "     To enable HDF5 & LeRobot dataset conversion, run:"
    echo "       ./install.sh"
fi

# ------------------------------------------------------------------------------
# 2. Verify .gemini/skills -> ../.agents/skills Symlink
# ------------------------------------------------------------------------------
echo ""
echo "[2/5] Verifying agent skills symlink (.gemini/skills -> ../.agents/skills)..."
mkdir -p "$WORKSPACE_ROOT/.gemini"
if [[ -e "$WORKSPACE_ROOT/.gemini/skills" && ! -L "$WORKSPACE_ROOT/.gemini/skills" ]]; then
    echo "[ERROR] .gemini/skills exists and is not a symlink; refusing to overwrite." >&2
    exit 1
fi
ln -sfn ../.agents/skills "$WORKSPACE_ROOT/.gemini/skills"

if [[ ! -f "$WORKSPACE_ROOT/.gemini/skills/pika-piper-control-guard/SKILL.md" ]]; then
    echo "[ERROR] Broken skills symlink: .gemini/skills/pika-piper-control-guard/SKILL.md not found." >&2
    exit 1
fi
echo "  [OK]      .gemini/skills -> $(readlink "$WORKSPACE_ROOT/.gemini/skills")"

# ------------------------------------------------------------------------------
# 3. Run Offline Pytest Suite (Zero Hardware / Zero Bytecode Litter)
# ------------------------------------------------------------------------------
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/quang_ws_bootstrap.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

echo ""
echo "[3/5] Running offline pytest suite..."
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT/Pika_Data/data_tools" \
    "$PYTHON_BIN" -m pytest -o "cache_dir=$tmp_dir/pytest_cache" -q

# ------------------------------------------------------------------------------
# 4. Run pikaumi CLI Doctor & Robot Registry Check
# ------------------------------------------------------------------------------
echo ""
echo "[4/5] Running standardized pika CLI doctor check..."
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$WORKSPACE_ROOT/src:$WORKSPACE_ROOT/Pika_Data/data_tools" \
    "$PYTHON_BIN" -m pikaumi.cli doctor --json-out "$tmp_dir/pika_doctor.json"

# ------------------------------------------------------------------------------
# 5. Run 5-Step Dual-Piper Policy Bridge Dry-Run Smoke Tests (pika + legacy)
# ------------------------------------------------------------------------------
echo ""
echo "[5/5] Running 5-step policy bridge smoke tests (--dry-run --mock-robot)..."
PYTHONDONTWRITEBYTECODE=1 ./pika policy-bridge \
    --robot piper_pika \
    --fps 30 \
    --steps 5 \
    --out "$tmp_dir/pikaumi_bridge_smoke.json"

PYTHONDONTWRITEBYTECODE=1 ./pika_data.sh policy-bridge \
    --dry-run \
    --mock-robot \
    --calibration-dir "$tmp_dir/mock_cal" \
    --fps 30 \
    --steps 5 \
    --out "$tmp_dir/bridge_smoke.json"

echo ""
echo "[SUCCESS] quang_ws (pikaumi + Pika_Data) bootstrap verification passed cleanly."
