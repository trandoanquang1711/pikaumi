#!/usr/bin/env bash
# install.sh — Standardized host setup & editable installation for pikaumi
# Safe by default: never touches live CAN buses or starts sensor streams.
set -euo pipefail

WORKSPACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_ROOT"

EXTRAS="dataset"
SKIP_UDEV=1
for arg in "$@"; do
    case "$arg" in
        --extras=*) EXTRAS="${arg#*=}" ;;
        --install-udev) SKIP_UDEV=0 ;;
        --help|-h)
            cat <<'EOF'
Usage: bash install.sh [--extras=dataset,sim,piper] [--install-udev]

Sets up .venv (via uv), installs pikaumi in editable mode,
links agent skills, and runs `pika doctor`.
EOF
            exit 0
            ;;
    esac
done

if command -v uv >/dev/null 2>&1 || [[ -x "$HOME/.local/bin/uv" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
    if [[ ! -d "$WORKSPACE_ROOT/.venv" ]]; then
        uv venv --python 3.12 "$WORKSPACE_ROOT/.venv"
    fi
    uv pip install --python "$WORKSPACE_ROOT/.venv/bin/python" -e ".[${EXTRAS}]"
    echo "[OK] Installed pikaumi in $WORKSPACE_ROOT/.venv (activate with: source .venv/bin/activate)"
else
    echo "[INFO] uv not found; using ./pika wrapper with system/existing virtualenv Python."
fi

if [[ "$SKIP_UDEV" -eq 0 ]]; then
    echo "[INFO] Installing Vive & Pika udev rules (requires sudo)..."
    sudo cp -f "$WORKSPACE_ROOT/Pika_Data/udev/81-vive.rules" /etc/udev/rules.d/81-vive.rules
    sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=tty --subsystem-match=video4linux
fi

mkdir -p "$WORKSPACE_ROOT/.gemini"
ln -sfn ../.agents/skills "$WORKSPACE_ROOT/.gemini/skills"

"$WORKSPACE_ROOT/pika" doctor
