"""
滚动回测脚本 - rolling_backtest.py
使用训练好的模型进行滚动窗口回测
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import matplotlib.pyplot as plt
import json
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


class RollingBacktestConfig:
    # Docker容器内路径
    BASE_DIR = "/app/code"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

    DATA_PATH = os.path.join(DATA_DIR, "train.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, "best_model.txt")
    FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.json")

    TARGET_COL = 'label'
    DATE_COL = '日期'
    CODE_COL = '股票代码'

    # 滚动窗口配置
    TEST_DAYS = 30
    STEP_DAYS = 30

    TOP_K = 5
    N_LAYERS = 5


def load_data_and_model():
    """加载数据和模型"""
    print(f"Loading data: {RollingBacktestConfig.DATA_PATH}")
    if not os.path.exists(RollingBacktestConfig.DATA_PATH):
        raise FileNotFoundError(f"Data not found: {RollingBacktestConfig.DATA_PATH}")
    if not os.path.exists(RollingBacktestConfig.MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {RollingBacktestConfig.MODEL_PATH}")

    df = pd.read_csv(RollingBacktestConfig.DATA_PATH)
    df[RollingBacktestConfig.DATE_COL] = pd.to_datetime(df[RollingBacktestConfig.DATE_COL])
    df = df.dropna(subset=[RollingBacktestConfig.TARGET_COL])

    model = lgb.Booster(model_file=RollingBacktestConfig.MODEL_PATH)

    with open(RollingBacktestConfig.FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    print(f"Data loaded: {len(df)} records")
    print(f"Date range: {df[RollingBacktestConfig.DATE_COL].min()} to {df[RollingBacktestConfig.DATE_COL].max()}")
    print(f"Model loaded, features: {len(feature_cols)}")

    return df, model, feature_cols


def evaluate_single_window(test_df, model, feature_cols, window_info):
    """评估单个时间窗口"""
    if len(test_df) == 0:
        return None

    # 处理缺失特征
    missing_cols = [col for col in feature_cols if col not in test_df.columns]
    for col in missing_cols:
        test_df[col] = 0

    # 生成预测
    X = test_df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    test_df['pred_score'] = model.predict(X)

    # 计算每日Top K收益和IC
    daily_returns_list = []
    daily_ic_list = []

    for test_date in sorted(test_df[RollingBacktestConfig.DATE_COL].unique()):
        day_data = test_df[test_df[RollingBacktestConfig.DATE_COL] == test_date]

        if len(day_data) < RollingBacktestConfig.TOP_K:
            continue

        # Top K 收益
        sorted_day = day_data.sort_values('pred_score', ascending=False)
        top_k_return = sorted_day.head(RollingBacktestConfig.TOP_K)[RollingBacktestConfig.TARGET_COL].mean()
        daily_returns_list.append(top_k_return)

        # IC
        if len(day_data) >= 10:
            ic = day_data['pred_score'].corr(day_data[RollingBacktestConfig.TARGET_COL], method='pearson')
            daily_ic_list.append(ic)

    if not daily_returns_list:
        return None

    daily_returns = pd.Series(daily_returns_list)
    daily_ic = pd.Series(daily_ic_list) if daily_ic_list else pd.Series([np.nan])

    return {
        'window_id': window_info['window_id'],
        'test_start': window_info['test_start'],
        'test_end': window_info['test_end'],
        'avg_return': daily_returns.mean(),
        'std_return': daily_returns.std(),
        'win_rate': (daily_returns > 0).mean(),
        'total_return': daily_returns.sum(),
        'avg_ic': daily_ic.mean(),
        'num_days': len(daily_returns),
    }


def rolling_backtest():
    """执行滚动窗口测试"""
    print("="*70)
    print("Rolling Window Backtest")
    print("="*70)

    # 1. 加载数据和模型
    df, model, feature_cols = load_data_and_model()

    # 2. 生成测试窗口
    all_dates = sorted(df[RollingBacktestConfig.DATE_COL].unique())
    n_dates = len(all_dates)

    windows = []
    start_idx = 0

    while True:
        test_end_idx = start_idx + RollingBacktestConfig.TEST_DAYS

        if test_end_idx > n_dates:
            break

        test_dates = all_dates[start_idx:test_end_idx]

        windows.append({
            'window_id': len(windows) + 1,
            'test_dates': test_dates,
            'test_start': test_dates[0],
            'test_end': test_dates[-1]
        })

        start_idx += RollingBacktestConfig.STEP_DAYS

    print(f"\nGenerated {len(windows)} test windows")
    print(f"  Window size: {RollingBacktestConfig.TEST_DAYS} days")
    print(f"  Step size: {RollingBacktestConfig.STEP_DAYS} days")

    # 3. 执行滚动测试
    results_list = []

    for window in tqdm(windows, desc="Backtesting"):
        try:
            test_df = df[df[RollingBacktestConfig.DATE_COL].isin(window['test_dates'])].copy()

            if len(test_df) == 0:
                continue

            result = evaluate_single_window(test_df, model, feature_cols, window)

            if result is not None:
                results_list.append(result)

        except Exception as e:
            print(f"\nWindow {window['window_id']} failed: {str(e)}")
            continue

    # 4. 汇总结果
    if not results_list:
        print("No successful test windows")
        return

    results_df = pd.DataFrame(results_list)

    print("\n" + "="*70)
    print("Backtest Results Summary")
    print("="*70)
    print(f"Successful windows: {len(results_df)}")
    print(f"\nAvg 5-day return: {results_df['avg_return'].mean():.4%} +/- {results_df['avg_return'].std():.4%}")
    print(f"Avg win rate: {results_df['win_rate'].mean():.2%} +/- {results_df['win_rate'].std():.2%}")
    print(f"Avg IC: {results_df['avg_ic'].mean():.4f} +/- {results_df['avg_ic'].std():.4f}")
    print(f"IC > 0 ratio: {(results_df['avg_ic'] > 0).mean():.2%}")

    # 5. 保存结果
    os.makedirs(RollingBacktestConfig.OUTPUT_DIR, exist_ok=True)
    results_summary = results_df[['window_id', 'test_start', 'test_end', 'avg_return', 'win_rate', 'avg_ic']].copy()
    results_summary.to_csv(os.path.join(RollingBacktestConfig.OUTPUT_DIR, 'backtest_results.csv'),
                          index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {os.path.join(RollingBacktestConfig.OUTPUT_DIR, 'backtest_results.csv')}")


if __name__ == "__main__":
    rolling_backtest()
