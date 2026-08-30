#!/usr/bin/env bash
set -euo pipefail

# Determine script root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"

echo "=== Installing Antigravity Plugin for Hermes Agent ==="
echo "Plugin source directory: ${SCRIPT_DIR}"
echo "Hermes home directory:   ${HERMES_HOME}"

# 1. Locate Hermes Python venv
HERMES_VENV=""
if [[ -d "${HERMES_HOME}/hermes-agent/venv" ]]; then
    HERMES_VENV="${HERMES_HOME}/hermes-agent/venv"
elif [[ -d "${HERMES_HOME}/venv" ]]; then
    HERMES_VENV="${HERMES_HOME}/venv"
elif command -v hermes &>/dev/null; then
    HERMES_BIN="$(command -v hermes)"
    if [[ -f "${HERMES_BIN}" ]]; then
        # Check if hermes is a script in a venv
        BIN_DIR="$(dirname "${HERMES_BIN}")"
        if [[ -f "${BIN_DIR}/python" ]]; then
            HERMES_VENV="$(dirname "${BIN_DIR}")"
        fi
    fi
fi

if [[ -n "${HERMES_VENV}" && -x "${HERMES_VENV}/bin/pip" ]]; then
    echo "Found Hermes Python environment at: ${HERMES_VENV}"
    echo "Installing hermes-antigravity-pi-port via pip entry-point..."
    "${HERMES_VENV}/bin/pip" install -q -e "${SCRIPT_DIR}"
    echo "✅ Pip package and hermes_agent.plugins entry point registered."
else
    echo "⚠️  Hermes virtual environment not found. Skipping pip install."
fi

# 2. Setup directory links for robust dual discovery
echo "Setting up plugin directories in ${HERMES_HOME}/plugins/..."
mkdir -p "${HERMES_HOME}/plugins/model-providers"
mkdir -p "${HERMES_HOME}/plugins"

ln -sfn "${SCRIPT_DIR}" "${HERMES_HOME}/plugins/model-providers/antigravity"
ln -sfn "${SCRIPT_DIR}" "${HERMES_HOME}/plugins/antigravity"
echo "✅ Symlinks created for model-providers and general plugin managers."

# 3. Run doctor diagnostic
echo ""
if [[ -n "${HERMES_VENV}" && -x "${HERMES_VENV}/bin/python" ]]; then
    "${HERMES_VENV}/bin/python" "${SCRIPT_DIR}/scripts/antigravity_doctor.py"
elif command -v uv &>/dev/null; then
    uv run python "${SCRIPT_DIR}/scripts/antigravity_doctor.py"
fi

echo ""
echo "🎉 Installation complete! Test with:"
echo "   hermes -z \"Say hello\" --provider antigravity -m gemini-3.7-flash"
