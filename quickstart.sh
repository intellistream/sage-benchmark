#!/bin/bash
# 🚀 SAGE Benchmark 快速初始化脚本
# 自动初始化所有 Git 子模块

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
DIM='\033[2m'

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  🚀 SAGE Benchmark 快速初始化${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 检查是否在正确的目录
if [ ! -f "$SCRIPT_DIR/pyproject.toml" ]; then
    echo -e "${RED}❌ 错误: 请在 sage-benchmark 根目录运行此脚本${NC}"
    exit 1
fi

# 检查 git 是否安装
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ 错误: 未安装 git${NC}"
    echo -e "${DIM}请安装 git: sudo apt-get install git${NC}"
    exit 1
fi

# 检查是否是 git 仓库
if [ ! -d "$SCRIPT_DIR/.git" ]; then
    echo -e "${RED}❌ 错误: 当前目录不是 git 仓库${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 环境检查通过${NC}"
echo ""

# 避免 Git LFS 自动拉取大文件
if [ -z "${GIT_LFS_SKIP_SMUDGE+x}" ]; then
    export GIT_LFS_SKIP_SMUDGE=1
    echo -e "${DIM}已设置 GIT_LFS_SKIP_SMUDGE=1 (跳过 LFS 大文件)${NC}"
fi

# 初始化子模块
echo -e "${BLUE}🔄 初始化 Git 子模块...${NC}"
echo -e "${DIM}将初始化以下子模块:${NC}"
echo -e "${DIM}  - src/sage/benchmark/benchmark_amm  (LibAMM)${NC}"
echo -e "${DIM}  - src/sage/benchmark/benchmark_anns (SAGE-DB-Bench)${NC}"
echo -e "${DIM}  - src/sage/data                      (sageData)${NC}"
echo ""

cd "$SCRIPT_DIR"

# 初始化子模块（并行加速）
if git submodule status | grep -q '^-'; then
    echo -e "${YELLOW}⚙️  初始化未初始化的子模块...${NC}"
    git submodule update --init --jobs 4
    echo -e "${GREEN}✓ 子模块初始化完成${NC}"
else
    echo -e "${GREEN}✓ 子模块已初始化${NC}"

    # 检查是否需要更新
    echo -e "${BLUE}🔄 检查子模块更新...${NC}"
    git submodule update --jobs 4
    echo -e "${GREEN}✓ 子模块已更新${NC}"
fi

echo ""

# 显示子模块状态
echo -e "${BLUE}📊 子模块状态:${NC}"
git submodule status | while read status; do
    commit=$(echo $status | awk '{print $1}')
    path=$(echo $status | awk '{print $2}')
    branch=$(echo $status | awk '{print $3}' | sed 's/[()]//g')

    if [[ $commit == -* ]]; then
        echo -e "${YELLOW}  ⚠️  $path - 未初始化${NC}"
    elif [[ $commit == +* ]]; then
        echo -e "${YELLOW}  ⚠️  $path - 未提交的更改${NC}"
    else
        echo -e "${GREEN}  ✓ $path - $branch${NC}"
    fi
done

# 安装 Git hooks
echo -e "${BLUE}🔧 安装 Git hooks...${NC}"
if [ -d "$SCRIPT_DIR/hooks" ]; then
    for hook_src in "$SCRIPT_DIR/hooks"/*; do
        hook_name=$(basename "$hook_src")
        hook_dst="$SCRIPT_DIR/.git/hooks/$hook_name"
        cp "$hook_src" "$hook_dst"
        chmod +x "$hook_dst"
        echo -e "${GREEN}  ✓ $hook_name${NC}"
    done
    echo -e "${GREEN}✓ Git hooks 已安装${NC}"
else
    echo -e "${YELLOW}  ⚠️  未找到 hooks/ 目录，跳过${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✨ 初始化完成！${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${DIM}下一步:${NC}"
echo -e "  1. 安装依赖: ${BLUE}pip install isage && pip install -e .${NC}"
echo -e "  2. 使用现有非-venv Python 环境（建议 conda）后再安装依赖${NC}"
echo ""
echo -e "${DIM}运行测试:${NC}"
echo -e "  ${BLUE}pytest tests/${NC}"
echo ""
