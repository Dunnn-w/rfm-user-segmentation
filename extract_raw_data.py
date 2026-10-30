# -*- coding: utf-8 -*-
"""
数据重建工具：从既有分析结果表 result_r/rfm_analysis_results.xlsx
提取原始 14 个用户字段，重建 user_personalized_features.xlsx。

背景：工程依赖的原始输入文件 user_personalized_features.xlsx 缺失，
但中间表完整保留了全部原始字段（仅追加了衍生列），故可无损重建。

用法：python extract_raw_data.py
"""
import os
import pandas as pd

# 输入：既有分析结果表（含原始14列 + 衍生列）
SOURCE = r"D:\新建文件夹1\量化学习\数分项目\result_r\rfm_analysis_results.xlsx"
# 输出：重建的原始数据（放到本工程的 data 目录）
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "data", "user_personalized_features.xlsx")

RAW_COLS = [
    "User_ID", "Age", "Gender", "Location", "Income", "Interests",
    "Last_Login_Days_Ago", "Purchase_Frequency", "Average_Order_Value",
    "Total_Spending", "Product_Category_Preference",
    "Time_Spent_on_Site_Minutes", "Pages_Viewed", "Newsletter_Subscription",
]


def main():
    df = pd.read_excel(SOURCE)
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"源表缺少列: {missing}")
    raw = df[RAW_COLS].copy()
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    raw.to_excel(OUTPUT, index=False)
    print(f"已重建原始数据 -> {OUTPUT}")
    print(f"规模: {raw.shape[0]} 行 x {raw.shape[1]} 列")
    print(f"缺失值: {raw.isnull().sum().sum()}")
    print(f"列: {list(raw.columns)}")


if __name__ == "__main__":
    main()