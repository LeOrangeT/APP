"""
数据预处理脚本 - data.py
处理原始股票数据，计算标签
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_and_process_data(csv_path, start_date=None, end_date=None, output_path='/app/code/data/F_train.csv'):
    """
    加载原始股票数据，进行基础数据预处理。

    参数:
        csv_path: 原始CSV文件路径
        start_date: 起始日期 (格式: 'YYYY-MM-DD')
        end_date: 结束日期 (格式: 'YYYY-MM-DD')
        output_path: 处理后数据保存路径

    返回:
        处理好的 DataFrame
    """
    print("="*70)
    print("Stock Data Preprocessing")
    print("="*70)

    # ==================== 1. 读取数据 ====================
    print(f"\n[1/5] Reading data: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  Raw data count: {len(df):,}")

    # ==================== 2. 验证必要列 ====================
    print("\n[2/5] Validating required columns...")
    required_columns = ['股票代码', '日期', '开盘', '最高', '最低', '收盘', '成交量']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    print(f"  Required columns validated")

    # ==================== 3. 日期处理与筛选 ====================
    print("\n[3/5] Processing dates...")
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')

    # 删除无效日期
    invalid_dates = df['日期'].isnull().sum()
    if invalid_dates > 0:
        print(f"  Removed {invalid_dates:,} invalid dates")
        df = df.dropna(subset=['日期'])

    print(f"  Date range: {df['日期'].min().date()} to {df['日期'].max().date()}")
    print(f"  Stock count: {df['股票代码'].nunique():,}")

    # 日期筛选
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df['日期'] >= start_date]
        print(f"  Start date: {start_date.date()}")

    if end_date:
        end_date = pd.to_datetime(end_date)
        df = df[df['日期'] <= end_date]
        print(f"  End date: {end_date.date()}")

    print(f"  Filtered data count: {len(df):,}")

    # ==================== 4. 数据排序与缺失值填充 ====================
    print("\n[4/5] Sorting and filling missing values...")

    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    numeric_columns = ['开盘', '最高', '最低', '收盘', '成交量']
    optional_columns = ['成交额', '涨跌幅', '换手率', '振幅', '涨跌额']

    all_numeric_cols = numeric_columns + [col for col in optional_columns if col in df.columns]

    for col in all_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(0)
        df[col] = df[col].replace([np.inf, -np.inf], 0)

    print(f"  Data type conversion completed")

    # ==================== 5. 计算未来5日收益率标签 ====================
    print("\n[5/5] Calculating 5-day return labels...")
    print("  Trading logic: T close -> T+1 open buy -> T+5 open sell")
    print("  Label formula: label = (Open_T+5 - Open_T+1) / Open_T+1")

    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    df['open_t1'] = df.groupby('股票代码')['开盘'].shift(-1)
    df['open_t5'] = df.groupby('股票代码')['开盘'].shift(-5)

    df['label_raw'] = (df['open_t5'] - df['open_t1']) / (df['open_t1'] + 1e-12)

    df['label'] = df.groupby('股票代码')['label_raw'].shift(1)

    for col in ['open_t1', 'open_t5', 'label_raw']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    if 'label' not in df.columns:
        raise ValueError("Label column not created")

    df_model = df.dropna(subset=['label']).copy()
    print(f"  Data count after removing null labels: {len(df_model):,}")

    # Winsorize处理
    print("\n[6/6] Winsorizing labels...")
    lower_quantile = df_model['label'].quantile(0.01)
    upper_quantile = df_model['label'].quantile(0.99)

    print(f"  Before - Min: {df_model['label'].min():.6f}, Max: {df_model['label'].max():.6f}")
    print(f"  Thresholds - Lower: {lower_quantile:.6f}, Upper: {upper_quantile:.6f}")

    df_model['label'] = np.clip(df_model['label'], lower_quantile, upper_quantile)

    print(f"  After - Min: {df_model['label'].min():.6f}, Max: {df_model['label'].max():.6f}")

    # 处理无穷值
    inf_count = np.isinf(df_model['label']).sum()
    if inf_count > 0:
        print(f"  Removed {inf_count} infinite labels")
        df_model.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_model = df_model.dropna(subset=['label']).copy()

    # 统计信息
    print(f"\n{'='*70}")
    print("Label Statistics:")
    print(f"{'='*70}")
    print(f"  Sample count: {len(df_model):,}")
    print(f"  Mean: {df_model['label'].mean():.6f}")
    print(f"  Std: {df_model['label'].std():.6f}")
    print(f"  Min: {df_model['label'].min():.6f}")
    print(f"  Max: {df_model['label'].max():.6f}")

    # ==================== 6. 保存处理结果 ====================
    print(f"\nSaving data to: {output_path}")

    cols_to_save = ['股票代码', '日期', '开盘', '最高', '最低', '收盘', '成交量', 'label']
    optional_save_cols = ['成交额', '涨跌幅', '换手率', '振幅', '涨跌额']
    for col in optional_save_cols:
        if col in df_model.columns:
            cols_to_save.append(col)

    existing_cols = [c for c in cols_to_save if c in df_model.columns]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_model[existing_cols].to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n{'='*70}")
    print("Preprocessing completed!")
    print(f"{'='*70}")
    print(f"Output file: {output_path}")
    print(f"Final data count: {len(df_model):,}")
    print(f"Stock count: {df_model['股票代码'].nunique():,}")

    return df_model


if __name__ == "__main__":
    # Docker容器内路径
    BASE_DIR = "/app/code"
    DATA_DIR = os.path.join(BASE_DIR, "data")

    CSV_PATH = os.path.join(DATA_DIR, "stock_data.csv")
    START_DATE = '2024-01-02'
    END_DATE = '2026-03-13'
    OUTPUT_PATH = os.path.join(DATA_DIR, "F_train.csv")

    try:
        processed_df = load_and_process_data(CSV_PATH, START_DATE, END_DATE, OUTPUT_PATH)
        print("\nData processing successful!")

    except Exception as e:
        print(f"\nProcessing failed: {str(e)}")
        import traceback
        traceback.print_exc()
