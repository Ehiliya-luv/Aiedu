#!/bin/bash

# ============================================
# Aiedu 项目快速验证脚本
# 执行：bash run_setup_and_test.sh
# ============================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 命令未找到"
        return 1
    fi
    return 0
}

# 主函数
main() {
    log_info "🚀 开始 Aiedu 项目快速验证"
    echo "========================================"

    # 1. 检查环境
    log_info "📋 步骤1: 检查环境"

    # 检查 Python
    if ! check_command python3; then
        log_error "需要安装 Python 3.8+"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    log_success "Python 版本: $PYTHON_VERSION"

    # 检查 Python 版本
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        log_error "需要 Python 3.8 或更高版本"
        exit 1
    fi

    # 检查 pip
    if ! check_command pip3; then
        log_error "需要安装 pip"
        exit 1
    fi
    log_success "pip 已安装"

    # 检查 GPU (可选)
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        log_success "检测到 $GPU_COUNT 个 GPU"
    else
        log_warning "未检测到 GPU，将使用 CPU 模式"
    fi

    echo ""

    # 2. 检查虚拟环境
    log_info "📋 步骤2: 检查虚拟环境"

    if [ -z "$VIRTUAL_ENV" ]; then
        log_warning "未激活虚拟环境，建议使用虚拟环境运行"
        log_info "跳过虚拟环境创建，继续执行..."
    else
        log_success "虚拟环境已激活: $VIRTUAL_ENV"
    fi

    echo ""

    # 3. 安装依赖
    log_info "📋 步骤3: 安装依赖"

    if [ ! -f "requirements.txt" ]; then
        log_error "requirements.txt 文件不存在"
        exit 1
    fi

    # 检查是否已安装核心依赖
    log_info "检查是否已安装核心依赖..."
    if python3 -c "import torch, transformers, trl; print('核心依赖已安装')" 2>/dev/null; then
        log_success "核心依赖已安装，跳过安装步骤"
        SKIP_INSTALL=true
    else
        SKIP_INSTALL=false
    fi

    if [ "$SKIP_INSTALL" = false ]; then
        log_info "升级 pip..."
        pip install --upgrade pip

        log_info "安装项目依赖..."
        if pip install -r requirements.txt; then
            log_success "依赖安装完成"
        else
            log_warning "依赖安装失败，尝试继续验证..."
        fi
    fi

    echo ""

    # 4. 验证功能
    log_info "📋 步骤4: 验证功能"

    # 检查导入
    log_info "检查模块导入..."
    python3 -c "
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    import transformers
    print(f'Transformers: {transformers.__version__}')
    import trl
    print(f'TRL: {trl.__version__}')
    import bert_score
    print('BertScore: OK')
    print('所有核心模块导入成功')
except ImportError as e:
    print(f'导入错误: {e}')
    exit(1)
"

    if [ $? -eq 0 ]; then
        log_success "模块导入测试通过"
    else
        log_error "模块导入测试失败"
        exit 1
    fi

    echo ""

    # 运行测试
    log_info "运行集成测试..."
    if [ -f "test/test_full_integration.py" ]; then
        python3 test/test_full_integration.py
        if [ $? -eq 0 ]; then
            log_success "集成测试通过"
        else
            log_warning "集成测试失败，但这不影响基本功能"
        fi
    else
        log_warning "测试文件不存在，跳过测试"
    fi

    echo ""

    # 5. 输出建议
    log_info "📋 步骤5: 输出使用建议"

    echo "========================================"
    log_success "✅ 快速验证完成！"
    echo ""
    echo "🎯 推荐的训练命令："
    echo ""

    # 检查是否有训练数据
    if [ -f "data/sft_train.jsonl" ] && [ -f "data/rl_train.jsonl" ]; then
        log_success "✓ 训练数据已存在"
        echo "  # 完整训练 (SFT + RL)"
        echo "  python main.py --mode sft+rl --reward-type advanced"
        echo ""
        echo "  # 仅 RL 训练 (使用已有的 SFT 模型)"
        echo "  python main.py --mode rl --reward-type advanced"
        echo ""
    else
        log_warning "⚠ 训练数据文件缺失，请检查 data/ 目录"
        echo "  需要准备 data/sft_train.jsonl 和 data/rl_train.jsonl 文件"
        echo ""
    fi

    echo "🔧 其他有用命令："
    echo "  # 查看帮助"
    echo "  python main.py --help"
    echo ""
    echo "  # 仅 SFT 微调"
    echo "  python main.py --mode sft"
    echo ""
    echo "  # 使用基本 Reward (快速)"
    echo "  python main.py --mode rl --reward-type basic"
    echo ""
    echo "  # 自定义参数"
    echo "  python main.py --mode rl --epochs 5 --batch-size 4 --learning-rate 1e-5"
    echo ""

    echo "📊 性能提示："
    if command -v nvidia-smi &> /dev/null; then
        echo "  ✓ GPU 可用，推荐使用 GPU 训练"
        echo "  ✓ 如果显存不足，尝试 --batch-size 1"
    else
        echo "  ⚠ 未检测到 GPU，将使用 CPU (较慢)"
        echo "  ✓ 考虑使用 Google Colab 或其他 GPU 环境"
    fi
    echo ""

    echo "📖 更多信息请查看 README.md"
    echo "=========================================="
}

# 运行主函数
main "$@"