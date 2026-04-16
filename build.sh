#!/bin/bash
# Docker镜像构建和导出脚本
# 使用方法: bash build.sh <队伍名称>

set -e

# 检查参数
if [ -z "$1" ]; then
    echo "使用方法: bash build.sh <队伍名称>"
    echo "示例: bash build.sh team_alpha"
    exit 1
fi

TEAM_NAME=$1
IMAGE_NAME="bdc2026"
OUTPUT_FILE="${TEAM_NAME}.tar"

echo "========================================"
echo "构建 Docker 镜像"
echo "========================================"
echo "镜像名称: ${IMAGE_NAME}"
echo "导出文件: ${OUTPUT_FILE}"
echo ""

# 构建镜像
echo ">>> 正在构建镜像..."
docker build -t ${IMAGE_NAME} .

# 检查构建结果
if [ $? -ne 0 ]; then
    echo "❌ 镜像构建失败！"
    exit 1
fi

echo ""
echo ">>> 正在导出镜像..."
docker save -o ${OUTPUT_FILE} ${IMAGE_NAME}

# 检查导出结果
if [ $? -ne 0 ]; then
    echo "❌ 镜像导出失败！"
    exit 1
fi

# 显示文件信息
echo ""
echo "========================================"
echo "✅ 构建完成！"
echo "========================================"
ls -lh ${OUTPUT_FILE}
echo ""
echo "提交文件: ${OUTPUT_FILE}"
