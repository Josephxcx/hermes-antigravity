#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
BOLD='\033[1m'

info()    { echo -e "${GREEN}==>${NC} ${BOLD}$*${NC}"; }
warn()    { echo -e "${YELLOW}  ! $*${NC}"; }
success() { echo -e "${GREEN}  ✓ $*${NC}"; }
fail()    { echo -e "${RED}  ✗ $*${NC}"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"

echo ""
echo -e "${BOLD}=== Hermes Antigravity Plugin Installer ===${NC}"
echo ""
info "Plugin source: ${SCRIPT_DIR}"
info "Hermes home:   ${HERMES_HOME}"
echo ""

# 1. Locate Hermes Python venv
VENV_PYTHON=""
VENV_PIP=""

if [[ -x "${HERMES_HOME}/hermes-agent/venv/bin/python" ]]; then
    VENV_PYTHON="${HERMES_HOME}/hermes-agent/venv/bin/python"
    VENV_PIP="${HERMES_HOME}/hermes-agent/venv/bin/pip"
    info "Found Hermes venv at: ${HERMES_HOME}/hermes-agent/venv"
elif [[ -x "${HERMES_HOME}/venv/bin/python" ]]; then
    VENV_PYTHON="${HERMES_HOME}/venv/bin/python"
    VENV_PIP="${HERMES_HOME}/venv/bin/pip"
    info "Found Hermes venv at: ${HERMES_HOME}/venv"
elif command -v hermes &>/dev/null; then
    HERMES_BIN="$(command -v hermes)"
    if [[ -f "${HERMES_BIN}" ]]; then
        BIN_DIR="$(dirname "${HERMES_BIN}")"
        if [[ -x "${BIN_DIR}/python" ]]; then
            VENV_PYTHON="${BIN_DIR}/python"
            VENV_PIP="${BIN_DIR}/pip"
            info "Found Hermes runtime at: ${BIN_DIR}"
        fi
    fi
fi

if [[ -z "${VENV_PYTHON}" ]]; then
    if command -v python3 &>/dev/null; then
        VENV_PYTHON="$(command -v python3)"
        VENV_PIP="$(command -v pip3 2>/dev/null || echo 'python3 -m pip')"
        warn "Hermes virtual environment not found — using system python3."
        warn "If you haven't installed Hermes yet, install it with:"
        warn "  curl -fsSL https://raw.githubusercontent.com/hermesagent/hermes/main/install.sh | bash"
    else
        fail "No Python runtime found. Please install Python 3.10+ or Hermes Agent first."
    fi
fi

echo ""
info "Installing hermes-antigravity into Python environment..."
$VENV_PIP install -q -e "${SCRIPT_DIR}"
success "Package installed"

# 2. Setup directory links for plugin discovery
info "Configuring plugin directory symlinks..."
mkdir -p "${HERMES_HOME}/plugins/model-providers"
mkdir -p "${HERMES_HOME}/plugins"

ln -sfn "${SCRIPT_DIR}" "${HERMES_HOME}/plugins/model-providers/antigravity"
ln -sfn "${SCRIPT_DIR}" "${HERMES_HOME}/plugins/antigravity"
success "Symlinks created in ${HERMES_HOME}/plugins"

# 3. Verification
echo ""
info "Verifying plugin installation..."
if $VENV_PYTHON -c "import hermes_antigravity; print('  Module loaded successfully')" 2>/dev/null; then
    success "Plugin verified"
else
    warn "Plugin import check encountered a warning or missing runtime context. Running doctor..."
    if [[ -f "${SCRIPT_DIR}/scripts/antigravity_doctor.py" ]]; then
        $VENV_PYTHON "${SCRIPT_DIR}/scripts/antigravity_doctor.py" || true
    fi
fi

echo ""
echo -e "${GREEN}${BOLD}=== Installation Complete! ===${NC}"
echo ""
echo -e "Next steps:"
echo -e "  ${BOLD}1. Authenticate:${NC}  hermes /antigravity.auth"
echo -e "  ${BOLD}2. Test a model:${NC}  hermes -z 'Hello!' --provider antigravity -m gemini-3.8-flash"
echo -e "  ${BOLD}3. Diagnostics:${NC}   hermes /antigravity.doctor"
echo ""
