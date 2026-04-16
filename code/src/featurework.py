"""
特征工程脚本 - featurework.py
生成技术指标特征
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings('ignore')


def generate_technical_features(df, windows=[5, 10, 20, 30, 60]):
    """
    基于个股历史行情生成技术面特征。
    """
    print("="*70)
    print("正在生成基础技术特征...")
    print("="*70)

    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    feature_dfs = []
    grouped = df.groupby('股票代码')

    for stock_code, group in tqdm(grouped, desc="计算特征", total=len(grouped)):
        group = group.copy()

        close = group['收盘']
        high = group['最高']
        low = group['最低']
        volume = group['成交量']
        open_price = group['开盘']

        epsilon = 1e-12

        for w in windows:
            # 1. 移动平均线 (MA)
            ma = close.rolling(window=w, min_periods=1).mean()
            group[f'ma_{w}'] = ma
            group[f'price_ma_ratio_{w}'] = (close - ma) / (ma + epsilon)

            # 2. 波动率 (Volatility)
            log_ret = np.log(close / close.shift(1))
            vol = log_ret.rolling(window=w, min_periods=1).std()
            group[f'volatility_{w}'] = vol

            # 3. 成交量相对比率
            vol_ma = volume.rolling(window=w, min_periods=1).mean()
            group[f'volume_ratio_{w}'] = volume / (vol_ma + epsilon)

            # 4. 动量因子
            group[f'momentum_{w}'] = (close / close.shift(w)) - 1

            # 5. 威廉指标位置
            high_n = high.rolling(window=w, min_periods=1).max()
            low_n = low.rolling(window=w, min_periods=1).min()
            group[f'price_position_{w}'] = (close - low_n) / (high_n - low_n + epsilon)

            # 6. 布林带指标
            bb_std = close.rolling(window=w, min_periods=1).std()
            bb_middle = ma
            group[f'bb_width_{w}'] = (2 * bb_std) / (bb_middle + epsilon)
            group[f'bb_position_{w}'] = (close - (bb_middle - 2*bb_std)) / (4*bb_std + epsilon)

            # 7. RSI相对强弱指标
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=w, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=w, min_periods=1).mean()
            rs = gain / (loss + epsilon)
            group[f'rsi_{w}'] = 100 - (100 / (1 + rs))

            # 8. MACD指标
            if w == 12:
                ema12 = close.ewm(span=12, min_periods=1).mean()
                ema26 = close.ewm(span=26, min_periods=1).mean()
                macd_line = ema12 - ema26
                signal_line = macd_line.ewm(span=9, min_periods=1).mean()
                group['macd'] = macd_line
                group['macd_signal'] = signal_line
                group['macd_hist'] = macd_line - signal_line

        # 9. KDJ指标
        low_9 = low.rolling(window=9, min_periods=1).min()
        high_9 = high.rolling(window=9, min_periods=1).max()
        rsv = (close - low_9) / (high_9 - low_9 + epsilon) * 100
        k_val = rsv.ewm(com=2, min_periods=1).mean()
        d_val = k_val.ewm(com=2, min_periods=1).mean()
        j_val = 3 * k_val - 2 * d_val
        group['kdj_k'] = k_val
        group['kdj_d'] = d_val
        group['kdj_j'] = j_val

        # 10. ATR平均真实波幅
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        group['atr_14'] = tr.rolling(window=14, min_periods=1).mean()
        group['atr_ratio'] = group['atr_14'] / (close + epsilon)

        # 11. 价格变化率
        group['return_1d'] = close.pct_change(1)
        group['return_3d'] = close.pct_change(3)
        group['return_5d'] = close.pct_change(5)
        group['return_10d'] = close.pct_change(10)

        # 12. 成交量变化
        group['volume_change_1d'] = volume.pct_change(1)
        group['volume_change_5d'] = volume.pct_change(5)

        # 13. 振幅指标
        group['amplitude'] = (high - low) / (close + epsilon)
        group['amplitude_ma5'] = group['amplitude'].rolling(window=5, min_periods=1).mean()

        # 14. 上下影线
        group['upper_shadow'] = (high - pd.concat([open_price, close], axis=1).max(axis=1)) / (close + epsilon)
        group['lower_shadow'] = (pd.concat([open_price, close], axis=1).min(axis=1) - low) / (close + epsilon)

        # 15. 量价关系
        group['price_volume_trend'] = (close.pct_change() * volume).rolling(window=10, min_periods=1).sum()
        group['on_balance_volume'] = (np.sign(close.diff()) * volume).cumsum()

        # 16. 相对强度
        group['intraday_return'] = (close - open_price) / (open_price + epsilon)

        feature_dfs.append(group)

    df_features = pd.concat(feature_dfs, ignore_index=True)
    return df_features


def apply_feature_lag(df, lag_days=1):
    """对特征进行滞后处理"""
    print(f"\n正在执行特征滞后处理 (Lag Days: {lag_days})...")

    static_cols = {'股票代码', '日期', 'label', '开盘', '最高', '最低', '收盘', '成交量',
                   '成交额', '涨跌幅', '换手率', '振幅', '涨跌额'}

    feature_cols = [col for col in df.columns if col not in static_cols and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]

    if not feature_cols:
        print("⚠️ 未找到需要滞后的新特征列")
        return df

    print(f"将对 {len(feature_cols)} 个特征列进行滞后处理...")

    df[feature_cols] = df.groupby('股票代码')[feature_cols].shift(lag_days)

    print("✓ 特征滞后处理完成")
    return df


def clean_and_prepare_final(df):
    """清理缺失值和无穷值"""
    print("\n正在清理数据...")
    original_len = len(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    cleaned_len = len(df)
    print(f"原始行数: {original_len:,} | 清理后行数: {cleaned_len:,} | 移除比例: {(1 - cleaned_len/original_len)*100:.2f}%")

    return df.reset_index(drop=True)


def save_dataset(df, output_path):
    """保存数据集"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"💾 数据集已保存至: {output_path}")


if __name__ == "__main__":
    # 使用相对路径
    BASE_DIR = "/app/code"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    TEMP_DIR = os.path.join(BASE_DIR, "temp")

    INPUT_PATH = os.path.join(DATA_DIR, "train.csv")
    OUTPUT_PATH = os.path.join(DATA_DIR, "train.csv")  # 覆盖原文件

    LAG_DAYS = 1
    WINDOWS = [5, 10, 20, 30, 60]

    try:
        if not os.path.exists(INPUT_PATH):
            raise FileNotFoundError(f"找不到输入文件: {INPUT_PATH}")

        print(f"📂 加载数据: {INPUT_PATH}")
        df = pd.read_csv(INPUT_PATH)
        df['日期'] = pd.to_datetime(df['日期'])

        df_feat = generate_technical_features(df, windows=WINDOWS)
        df_lagged = apply_feature_lag(df_feat, lag_days=LAG_DAYS)
        df_final = clean_and_prepare_final(df_lagged)
        save_dataset(df_final, OUTPUT_PATH)

        print("\n✅ 特征工程全部完成！")
        print(f"📊 最终样本数: {len(df_final):,}")
        print(f"📊 特征总数: {len(df_final.columns) - 3}")

    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
