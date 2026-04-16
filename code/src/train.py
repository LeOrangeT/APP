"""
训练脚本 - train.py
使用LightGBM LambdaRank训练股票排序模型
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import warnings
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings('ignore')

# ======================
# 1. 配置参数（使用相对路径）
# ======================
class Config:
    # 基础路径
    BASE_DIR = "/app/code"

    # 数据路径
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DATA_PATH = os.path.join(DATA_DIR, "train.csv")

    # 模型输出路径
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best_model.txt")
    SCALER_SAVE_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
    FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_names.json")

    # 输出路径
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    TEMP_DIR = os.path.join(BASE_DIR, "temp")

    # 数据列名
    TARGET_COL = 'label'
    DATE_COL = '日期'
    CODE_COL = '股票代码'

    # LightGBM 超参数
    BASE_PARAMS = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.03,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 30,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'max_depth': 6,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1,
        'eval_at': [1, 3, 5, 10],
        'label_gain': [0, 1, 3, 7, 15]
    }
    NUM_ROUNDS = 1500
    EARLY_STOP = 100
    NUM_BINS = 5

    # 数据集划分比例
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.85

# ======================
# 2. 辅助函数
# ======================

def ensure_dirs():
    """确保所有必要的输出目录存在"""
    for dir_path in [Config.MODEL_DIR, Config.OUTPUT_DIR, Config.TEMP_DIR]:
        os.makedirs(dir_path, exist_ok=True)


def load_and_prepare_data():
    """加载数据并筛选特征"""
    print(f"📂 正在加载数据: {Config.DATA_PATH}")
    if not os.path.exists(Config.DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {Config.DATA_PATH}")

    df = pd.read_csv(Config.DATA_PATH)

    # 处理日期
    if Config.DATE_COL in df.columns:
        df[Config.DATE_COL] = pd.to_datetime(df[Config.DATE_COL])

    # 处理无穷值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 筛选数值型特征
    exclude_keywords = [
        Config.CODE_COL, Config.DATE_COL, Config.TARGET_COL, 'label_int',
        '开盘', '最高', '最低', '收盘', '成交量', '成交额',
        '涨跌幅', '换手率', '振幅', '涨跌额', 'open', 'high', 'low', 'close', 'volume'
    ]

    feature_cols = []
    for col in df.columns:
        if not any(k in col for k in exclude_keywords) and df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            feature_cols.append(col)

    print(f"✅ 数据加载完成。样本数: {len(df)}, 特征数: {len(feature_cols)}")
    return df, feature_cols


def discretize_labels(df):
    """将连续标签离散化为等级"""
    print(f"🏷️ 正在将标签离散化为 {Config.NUM_BINS} 个等级...")

    def rank_group(group):
        try:
            group['label_int'] = pd.qcut(group[Config.TARGET_COL], q=Config.NUM_BINS, labels=False, duplicates='drop')
        except ValueError:
            group['label_int'] = pd.cut(group[Config.TARGET_COL].rank(pct=True), bins=Config.NUM_BINS, labels=False)
        return group

    df = df.groupby(Config.DATE_COL, group_keys=False).apply(rank_group)
    df['label_int'] = df['label_int'].fillna(0).astype(int)
    return df


# ======================
# 3. 核心训练逻辑
# ======================

def train_ranking_model():
    ensure_dirs()

    # 1. 数据准备
    df, feature_cols = load_and_prepare_data()
    if len(df) == 0:
        raise ValueError("数据为空")

    df = discretize_labels(df)

    # 2. 时间序列划分
    dates = sorted(df[Config.DATE_COL].unique())
    n_dates = len(dates)
    train_end_idx = int(n_dates * Config.TRAIN_RATIO)
    val_end_idx = int(n_dates * Config.VAL_RATIO)

    train_dates = dates[:train_end_idx]
    val_dates = dates[train_end_idx:val_end_idx]

    train_df = df[df[Config.DATE_COL].isin(train_dates)].sort_values(Config.DATE_COL)
    val_df = df[df[Config.DATE_COL].isin(val_dates)].sort_values(Config.DATE_COL)

    print(f"\n🚀 开始训练模型...")
    print(f"   训练集日期范围: {train_dates[0]} ~ {train_dates[-1]}")
    print(f"   验证集日期范围: {val_dates[0]} ~ {val_dates[-1]}")
    print(f"   使用特征数: {len(feature_cols)}")

    # 3. 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_val = scaler.transform(val_df[feature_cols].values)

    # 4. 构建 LightGBM Dataset
    train_data = lgb.Dataset(
        data=X_train,
        label=train_df['label_int'].values,
        group=train_df.groupby(Config.DATE_COL).size().values,
        feature_name=feature_cols,
        free_raw_data=False
    )

    val_data = lgb.Dataset(
        data=X_val,
        label=val_df['label_int'].values,
        group=val_df.groupby(Config.DATE_COL).size().values,
        feature_name=feature_cols,
        free_raw_data=False
    )

    # 5. 训练模型
    final_model = lgb.train(
        Config.BASE_PARAMS,
        train_data,
        num_boost_round=Config.NUM_ROUNDS,
        valid_sets=[val_data],
        valid_names=['validation'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=Config.EARLY_STOP),
            lgb.log_evaluation(period=100)
        ]
    )

    # 6. 输出最佳结果
    best_iter = final_model.best_iteration
    best_scores = final_model.best_score.get('validation', {})
    target_metric = 'ndcg@5' if 'ndcg@5' in best_scores else next(iter(best_scores))
    best_final_score = best_scores[target_metric]

    print(f"\n✅ 训练完成！")
    print(f"   Best epoch: {best_iter}")
    print(f"   Best {target_metric}: {best_final_score:.6f}")

    # 7. 保存模型
    final_model.save_model(Config.MODEL_SAVE_PATH)
    print(f"💾 模型已保存: {Config.MODEL_SAVE_PATH}")

    # 保存 Scaler
    joblib.dump(scaler, Config.SCALER_SAVE_PATH)
    print(f"💾 Scaler 已保存: {Config.SCALER_SAVE_PATH}")

    # 保存特征列表
    with open(Config.FEATURE_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=4)
    print(f"💾 特征列表已保存: {Config.FEATURE_LIST_PATH}")


# ======================
# 4. 入口
# ======================

if __name__ == "__main__":
    train_ranking_model()
