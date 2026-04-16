#!/bin/bash
# 初始化脚本 - 环境检查和目录创建

set -e

echo "========================================"
echo "初始化环境"
echo "========================================"

# 创建必要的目录
mkdir -p /app/code/data
mkdir -p /app/code/model
mkdir -p /app/code/output
mkdir -p /app/code/temp

# 检查Python版本
echo "Python版本:"
python --version

# 检查依赖
echo ""
echo "检查依赖包..."
python -c "import pandas; print(f'pandas: {pandas.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import lightgbm; print(f'lightgbm: {lightgbm.__version__}')"

# 检查数据文件
echo ""
echo "检查数据文件..."
if [ -f "/app/code/data/train.csv" ]; then
    echo "✓ train.csv 存在"
else
    echo "✗ train.csv 不存在"
fi

if [ -f "/app/code/data/test.csv" ]; then
    echo "✓ test.csv 存在"
else
    echo "✗ test.csv 不存在"
fi

echo ""
echo "========================================"
echo "初始化完成"
echo "========================================"
