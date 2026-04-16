# 沪深300股票5日收益率排序预测系统

基于LightGBM LambdaRank的股票排序预测模型，预测沪深300成分股的5日收益率并进行每日选股排名。

---

## 项目结构

```
app/
├── code/                          # 运行代码（docker内）
│   ├── src/
│   │   ├── featurework.py         # 特征工程
│   │   ├── train.py               # 模型训练
│   │   └── test.py                # 推理脚本
│   ├── data/                      # 数据目录（docker-compose挂载）
│   │   ├── train.csv              # 训练数据
│   │   └── test.csv               # 测试数据
│   ├── model/                     # 模型目录（docker内）
│   ├── output/                    # 输出目录（docker-compose挂载）
│   │   └── result.csv             # 推理结果
│   ├── temp/                      # 中间结果（docker-compose挂载）
│   ├── init.sh                    # 初始化脚本（必选）
│   ├── train.sh                   # 训练脚本（必选）
│   └── test.sh                    # 测试脚本（必选）
├── readme.md                      # 说明文档（必选）
├── requirements.txt               # Python依赖
├── Dockerfile                     # Docker构建文件
└── docker-compose.yml             # Docker编排文件
```

---

## 环境配置

### Python版本
- Python 3.9

### 依赖版本

| 依赖包 | 版本要求 | 用途 |
|--------|----------|------|
| pandas | >=1.5.0 | 数据处理 |
| numpy | >=1.24.0 | 数值计算 |
| lightgbm | >=4.0.0 | 排序学习模型 |
| tqdm | >=4.65.0 | 进度条 |
| scikit-learn | >=1.3.0 | 数据预处理、评估指标 |
| joblib | >=1.3.0 | 模型序列化 |

---

## Docker 使用说明

### 1. 构建镜像

**Linux/Mac:**
```bash
bash build.sh 队伍名称
```

**Windows:**
```cmd
build.bat 队伍名称
```

构建完成后生成 `队伍名称.tar` 文件。

### 2. 手动构建命令

```bash
# 构建镜像（镜像名必须为 bdc2026）
docker build -t bdc2026 .

# 导出镜像
docker save -o 队伍名称.tar bdc2026
```

### 3. 主办方加载和运行

```bash
# 加载镜像
docker load -i 队伍名称.tar

# 运行（使用主办方下发的docker-compose.yml）
docker-compose up
```

### 4. 目录挂载说明

| 容器路径 | 说明 | 挂载方式 |
|----------|------|----------|
| `/app/code/data` | 训练/测试数据 | docker-compose 挂载 |
| `/app/code/output` | 输出结果 | docker-compose 挂载 |
| `/app/code/temp` | 中间结果 | docker-compose 挂载 |
| `/app/code/model` | 模型文件 | docker 内部存储 |

---

## 数据

### 数据格式

**train.csv / test.csv 列要求:**

| 列名 | 类型 | 说明 |
|------|------|------|
| 股票代码 | string | 股票代码（如 sh.600000） |
| 日期 | datetime | 交易日期 |
| 开盘 | float | 开盘价 |
| 最高 | float | 最高价 |
| 最低 | float | 最低价 |
| 收盘 | float | 收盘价 |
| 成交量 | float | 成交量 |
| label | float | 5日收益率（仅train.csv需要） |
| 特征列 | float | 技术指标特征 |

### 输出格式

**output/result.csv 格式:**

| 列名 | 类型 | 说明 |
|------|------|------|
| stock_id | string | 股票代码 |
| weight | float | 权重（和为1） |

---

## 算法

### 整体思路

本项目采用 **排序学习(Learning to Rank)** 方法：

1. **预测目标**: 预测股票 **T+1日开盘买入 → T+5日开盘卖出** 的收益率
2. **标签定义**: `label = (Open_T+5 - Open_T+1) / Open_T+1`
3. **模型**: LightGBM LambdaRank，优化NDCG指标

### 网络结构

```
LightGBM LambdaRank
├── boosting_type: GBDT
├── num_leaves: 31
├── max_depth: 6
├── learning_rate: 0.03
├── feature_fraction: 0.7
├── bagging_fraction: 0.7
├── lambda_l1: 0.5
├── lambda_l2: 0.5
├── objective: lambdarank
├── metric: ndcg
└── eval_at: [1, 3, 5, 10]
```

### 损失函数

LambdaRank损失函数，直接优化NDCG指标。

### 特征工程

- 移动平均线 (MA): 5/10/20/30/60日
- 波动率、RSI、MACD、KDJ、ATR
- 价格变化率、量价关系指标
- 特征滞后处理（lag_days=1）

---

## 训练流程

运行 `train.sh` 或：
```bash
cd /app/code
python src/train.py
```

### train.py 详细步骤

#### Step 1: 配置参数定义
定义数据路径、模型参数、LightGBM超参数。

#### Step 2: 创建输出目录
创建 model/、output/、temp/ 目录。

#### Step 3: 加载数据
读取 `/app/code/data/train.csv`，处理日期列和无穷值。

#### Step 4: 标签离散化
使用 pd.qcut 将label离散化为5个等级。

#### Step 5: 数据集划分
按时间序列划分：70%训练集，15%验证集。

#### Step 6: 特征标准化
使用 StandardScaler 进行Z-score标准化。

#### Step 7: 构建LightGBM Dataset
按日期分组构建排序学习Dataset。

#### Step 8: 训练模型
使用早停机制训练LightGBM模型。

#### Step 9: 保存模型资产
保存模型、Scaler、特征列表到 `/app/code/model/`。

---

## 推理流程

运行 `test.sh` 或：
```bash
cd /app/code
python src/test.py
```

### test.py 详细步骤

#### Step 1: 加载模型资产
加载模型、Scaler、特征名称列表。

#### Step 2: 加载测试数据
读取 `/app/code/data/test.csv`。

#### Step 3: 特征标准化
使用训练时的Scaler进行标准化。

#### Step 4: 模型预测
生成预测得分。

#### Step 5: 取Top 5股票
按得分排序，取前5只股票。

#### Step 6: 计算权重
使用Softmax计算权重（和为1）。

#### Step 7: 保存结果
保存到 `/app/code/output/result.csv`。

---

## 其他注意事项

### 验证数据划分

采用时间序列划分，确保无数据泄露：

```
|<── 训练集 (70%) ──>|<─ 验证集 (15%) ─>|<─ 测试集 (15%) ─>|
```

### 前视偏差避免

1. 特征滞后1天
2. 标签附着在T+1日
3. 严格时间顺序划分

### 模型评估指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| NDCG@5 | 排序质量 | > 0.7 |
| IC | 预测与实际相关性 | > 0.03 |

---

## 免责声明

本项目仅供学习和研究使用，不构成投资建议。项目已经通过Apache License 2.0协议开源，详见https://github.com/LeOrangeT/APP

