"""
推理脚本 - predict.py
加载训练好的模型，对数据进行预测
"""
import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import warnings

warnings.filterwarnings('ignore')

# ======================
# 1. 配置参数（Docker容器内路径）
# ======================
class PredictConfig:
    # 基础路径
    BASE_DIR = "/app/code"

    # 数据路径
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DATA_PATH = os.path.join(DATA_DIR, "test.csv")

    # 模型路径
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    MODEL_PATH = os.path.join(MODEL_DIR, "best_model.txt")
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
    FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_names.json")

    # 输出路径
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "result.csv")

    # 数据列名
    DATE_COL = '日期'
    CODE_COL = '股票代码'

# ======================
# 2. 核心逻辑
# ======================

def load_assets():
    """加载训练好的模型、Scaler和特征列表"""
    print("Loading model assets...")
    print(f"Model directory: {PredictConfig.MODEL_DIR}")

    # 检查目录是否存在
    if not os.path.exists(PredictConfig.MODEL_DIR):
        raise FileNotFoundError(f"Model directory not found: {PredictConfig.MODEL_DIR}")

    # 列出文件
    files = os.listdir(PredictConfig.MODEL_DIR)
    print(f"Directory contents: {files}")

    # 1. 加载特征列表
    if not os.path.exists(PredictConfig.FEATURE_LIST_PATH):
        json_files = [f for f in files if f.endswith('.json')]
        if json_files:
            print(f"Using JSON file: {json_files[0]}")
            PredictConfig.FEATURE_LIST_PATH = os.path.join(PredictConfig.MODEL_DIR, json_files[0])
        else:
            raise FileNotFoundError(f"Feature list not found: {PredictConfig.FEATURE_LIST_PATH}")

    with open(PredictConfig.FEATURE_LIST_PATH, 'r', encoding='utf-8') as f:
        feature_cols = json.load(f)

    # 2. 加载 Scaler
    if not os.path.exists(PredictConfig.SCALER_PATH):
        pkl_files = [f for f in files if f.endswith('.pkl')]
        if pkl_files:
            print(f"Using PKL file: {pkl_files[0]}")
            PredictConfig.SCALER_PATH = os.path.join(PredictConfig.MODEL_DIR, pkl_files[0])
        else:
            raise FileNotFoundError(f"Scaler not found: {PredictConfig.SCALER_PATH}")
    scaler = joblib.load(PredictConfig.SCALER_PATH)

    # 3. 加载模型
    if not os.path.exists(PredictConfig.MODEL_PATH):
        model_files = [f for f in files if f.endswith('.txt') or f.endswith('.lgb')]
        if model_files:
            print(f"Using model file: {model_files[0]}")
            PredictConfig.MODEL_PATH = os.path.join(PredictConfig.MODEL_DIR, model_files[0])
        else:
            raise FileNotFoundError(f"Model not found: {PredictConfig.MODEL_PATH}")

    model = lgb.Booster(model_file=PredictConfig.MODEL_PATH)

    print(f"Assets loaded successfully (features: {len(feature_cols)})")
    return model, scaler, feature_cols


def get_latest_data(feature_cols):
    """加载数据并提取最新一个交易日的股票数据"""
    print(f"Loading data: {PredictConfig.DATA_PATH}")
    if not os.path.exists(PredictConfig.DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {PredictConfig.DATA_PATH}")

    df = pd.read_csv(PredictConfig.DATA_PATH)

    # 处理日期
    if PredictConfig.DATE_COL in df.columns:
        df[PredictConfig.DATE_COL] = pd.to_datetime(df[PredictConfig.DATE_COL])

    # 处理无穷值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 获取最新交易日
    latest_date = df[PredictConfig.DATE_COL].max()
    print(f"Latest trading date: {latest_date}")

    # 筛选最新日期的数据
    latest_df = df[df[PredictConfig.DATE_COL] == latest_date].copy()

    if latest_df.empty:
        raise ValueError(f"No data for latest date: {latest_date}")

    # 检查特征完整性，缺失的用0填充
    missing_cols = [col for col in feature_cols if col not in latest_df.columns]
    if missing_cols:
        print(f"Warning: Missing features filled with 0: {missing_cols[:5]}...")
        for col in missing_cols:
            latest_df[col] = 0

    print(f"Candidate stocks: {len(latest_df)}")
    return latest_df


def predict_and_save(model, scaler, latest_df, feature_cols):
    """执行标准化、预测、排序并保存结果"""
    print("Running inference...")

    # 1. 提取特征矩阵
    X_latest = latest_df[feature_cols].values

    # 2. 处理缺失值
    X_latest = np.nan_to_num(X_latest, nan=0.0)

    # 3. 特征标准化
    X_scaled = scaler.transform(X_latest)

    # 4. 模型预测
    scores = model.predict(X_scaled)

    # 5. 组装结果
    result_df = pd.DataFrame({
        'stock_id': latest_df[PredictConfig.CODE_COL].values,
        'score': scores
    })

    # 6. 排序
    result_df = result_df.sort_values(by='score', ascending=False)

    # 7. 取前5
    top5_df = result_df.head(5).reset_index(drop=True)

    # 8. 固定权重
    top5_df['weight'] = 0.2

    # 9. 保留列
    final_output = top5_df[['stock_id', 'weight']]

    # 10. 保存
    os.makedirs(PredictConfig.OUTPUT_DIR, exist_ok=True)
    final_output.to_csv(PredictConfig.OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')

    print("\nInference completed! Top 5 stocks:")
    print(final_output.to_string(index=False))
    print(f"\nResult saved to: {PredictConfig.OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    try:
        model, scaler, feature_cols = load_assets()
        latest_df = get_latest_data(feature_cols)
        predict_and_save(model, scaler, latest_df, feature_cols)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
