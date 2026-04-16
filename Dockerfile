# 沪深300股票预测系统 Docker镜像
# 镜像名称: bdc2026
# 基于Python 3.9 slim版本

FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制代码文件（docker内）
COPY code/ ./code/

# 复制README
COPY readme.md ./readme.md

# 创建目录结构
# data, output, temp 将通过docker-compose挂载
# model 在docker内
RUN mkdir -p /app/code/data \
    && mkdir -p /app/code/model \
    && mkdir -p /app/code/output \
    && mkdir -p /app/code/temp

# 给脚本添加执行权限
RUN chmod +x /app/code/init.sh \
    && chmod +x /app/code/train.sh \
    && chmod +x /app/code/test.sh

# 设置默认命令
CMD ["bash", "/app/code/init.sh"]
