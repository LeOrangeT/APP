"""
推理脚本 - test.py
加载训练好的模型，对测试数据进行预测，输出结果到 output/result.csv
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
# 1. 配置参数
# ======================
class Config:
    BASE_DIR = "/app/code"

    DATA_DIR = os.path.join(BASE_DIR, "data")
    TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")

    MODEL_DIR = os.path.join(BASE_DIR, "model")
    MODEL_PATH = os.path.join(MODEL_DIR, "best_model.txt")
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
    FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_names.json")

    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "result.csv")

    DATE_COL = '日期'
    CODE_COL = '股票代码'


# ======================
# 2. 辅助函数
# ======================

def load_assets():
    """加载模型、Scaler和特征列表"""
    print("Loading model assets...")

    with open(Config.FEATURE_LIST_PATH, 'r', encoding='utf-8') as f:
        feature_cols = json.load(f)

    scaler = joblib.load(Config.SCALER_PATH)
    model = lgb.Booster(model_file=Config.MODEL_PATH)

    print(f"Assets loaded (features: {len(feature_cols)})")
    return model, scaler, feature_cols


def load_test_data(feature_cols):
    """加载测试数据"""
    print(f"Loading test data: {Config.TEST_DATA_PATH}")

    if not os.path.exists(Config.TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data not found: {Config.TEST_DATA_PATH}")

    df = pd.read_csv(Config.TEST_DATA_PATH)

    if Config.DATE_COL in df.columns:
        df[Config.DATE_COL] = pd.to_datetime(df[Config.DATE_COL])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 检查并填充缺失特征
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing features filled with 0: {missing_cols[:5]}...")
        for col in missing_cols:
            df[col] = 0

    print(f"Test data loaded: {len(df)} rows")
    return df, feature_cols


def predict_and_save(model, scaler, df, feature_cols):
    """执行预测并保存结果"""
    print("Running inference...")

    # 提取特征
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)

    # 标准化
    X_scaled = scaler.transform(X)

    # 预测
    scores = model.predict(X_scaled)
    df['pred_score'] = scores

    # 获取最新日期
    if Config.DATE_COL in df.columns:
        latest_date = df[Config.DATE_COL].max()
        latest_df = df[df[Config.DATE_COL] == latest_date].copy()
    else:
        latest_df = df.copy()

    # 按分数排序取Top 5
    top5 = latest_df.nlargest(5, 'pred_score')

    # 计算Softmax权重
    top5_scores = top5['pred_score'].values
    exp_scores = np.exp(top5_scores - np.max(top5_scores))
    weights = exp_scores / exp_scores.sum()

    # 构建输出
    result_df = pd.DataFrame({
        'stock_id': top5[Config.CODE_COL].values,
        'weight': weights
    })

    # 保存
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    result_df.to_csv(Config.OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')

    print("\nInference completed! Top 5 stocks:")
    print(result_df.to_string(index=False))
    print(f"\nResult saved to: {Config.OUTPUT_CSV_PATH}")


# ======================
# 3. 主函数
# ======================

def main():
    try:
        model, scaler, feature_cols = load_assets()
        df, feature_cols = load_test_data(feature_cols)
        predict_and_save(model, scaler, df, feature_cols)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
