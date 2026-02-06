#!/bin/bash
# sage-benchmark Quickstart Script
# Sets up development environment and git hooks

set -e

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Print banner
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${BLUE}   _____ ___   ________________    ____                  __  ${NC}"
echo -e "${BOLD}${BLUE}  / ___//   | / ____/ ____/  _/   / __ )___  ____  _____/ /_ ${NC}"
echo -e "${BOLD}${BLUE}  \\__ \\/ /| |/ / __/ __/  / /_____/ __  / _ \\/ __ \\/ ___/ __ \\ ${NC}"
echo -e "${BOLD}${BLUE} ___/ / ___ / /_/ / /____/ /_____/ /_/ /  __/ / / / /__/ / / /${NC}"
echo -e "${BOLD}${BLUE}/____/_/  |_\\____/_____/___/    /_____/\\___/_/ /_/\\___/_/ /_/ ${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}SAGE Benchmark Quickstart Setup${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Detect project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo -e "${BLUE}📂 Project root: ${NC}$PROJECT_ROOT"
echo ""

# Parse command line arguments
DEV_MODE=false
SKIP_HOOKS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            DEV_MODE=true
            shift
            ;;
        --skip-hooks)
            SKIP_HOOKS=true
            shift
            ;;
        --help)
            echo "Usage: ./quickstart.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev          Install development dependencies"
            echo "  --skip-hooks   Skip git hooks installation"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Step 1: Install git hooks (unless --skip-hooks)
if [ "$SKIP_HOOKS" = false ]; then
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}${BOLD}Step 1: Installing Git Hooks${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    HOOKS_DIR="$PROJECT_ROOT/.git/hooks"
    TEMPLATE_DIR="$PROJECT_ROOT/hooks"

    if [ ! -d "$HOOKS_DIR" ]; then
        echo -e "${RED}✗ Git repository not initialized${NC}"
        echo -e "${YELLOW}Run: git init${NC}"
        exit 1
    fi

    # Install pre-commit hook
    if [ -f "$TEMPLATE_DIR/pre-commit" ]; then
        cp "$TEMPLATE_DIR/pre-commit" "$HOOKS_DIR/pre-commit"
        chmod +x "$HOOKS_DIR/pre-commit"
        echo -e "${GREEN}✓ Installed pre-commit hook${NC}"
    fi

    # Install pre-push hook
    if [ -f "$TEMPLATE_DIR/pre-push" ]; then
        cp "$TEMPLATE_DIR/pre-push" "$HOOKS_DIR/pre-push"
        chmod +x "$HOOKS_DIR/pre-push"
        echo -e "${GREEN}✓ Installed pre-push hook${NC}"
    fi

    echo ""
else
    echo -e "${YELLOW}⊘ Skipping git hooks installation${NC}"
    echo ""
fi

# Step 2: Install Python dependencies
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}${BOLD}Step 2: Installing Python Dependencies${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo -e "${RED}✗ Python 3.10+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python version check passed: $PYTHON_VERSION${NC}"
echo ""

# Install package
if [ "$DEV_MODE" = true ]; then
    echo -e "${BLUE}Installing with development dependencies...${NC}"
    pip install -e ".[dev]"
else
    echo -e "${BLUE}Installing core package...${NC}"
    pip install -e .
fi

echo ""
echo -e "${GREEN}✓ Python dependencies installed${NC}"
echo ""

# Step 3: Check SAGE installation
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}${BOLD}Step 3: Checking SAGE Dependencies${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if python3 -c "import sage.common" 2>/dev/null; then
    echo -e "${GREEN}✓ sage.common (L1) found${NC}"
else
    echo -e "${RED}✗ sage.common (L1) not found${NC}"
    echo -e "${YELLOW}Please install SAGE first: pip install isage-common${NC}"
fi

if python3 -c "import sage.kernel" 2>/dev/null; then
    echo -e "${GREEN}✓ sage.kernel (L3) found${NC}"
else
    echo -e "${RED}✗ sage.kernel (L3) not found${NC}"
    echo -e "${YELLOW}Please install SAGE: pip install isage-kernel${NC}"
fi

if python3 -c "import sage.libs" 2>/dev/null; then
    echo -e "${GREEN}✓ sage.libs (L3) found${NC}"
else
    echo -e "${RED}✗ sage.libs (L3) not found${NC}"
    echo -e "${YELLOW}Please install SAGE: pip install isage-libs${NC}"
fi

if python3 -c "import sage.middleware" 2>/dev/null; then
    echo -e "${GREEN}✓ sage.middleware (L4) found${NC}"
else
    echo -e "${RED}✗ sage.middleware (L4) not found${NC}"
    echo -e "${YELLOW}Please install SAGE: pip install isage-middleware${NC}"
fi

echo ""

# Step 4: Summary
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}✓ Setup Complete!${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BOLD}Next Steps:${NC}"
echo -e "  1. Run experiments: ${CYAN}python -m sage_benchmark${NC}"
echo -e "  2. View configuration: ${CYAN}ls config/\*.yaml${NC}"
echo -e "  3. Check documentation: ${CYAN}cat README.md${NC}"
echo ""
echo -e "${BOLD}Development Commands:${NC}"
echo -e "  • Run tests: ${CYAN}pytest${NC}"
echo -e "  • Format code: ${CYAN}ruff format .${NC}"
echo -e "  • Lint code: ${CYAN}ruff check .${NC}"
echo ""
echo -e "${GREEN}Happy benchmarking! 🚀${NC}"
echo ""
