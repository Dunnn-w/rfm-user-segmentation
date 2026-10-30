import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

def load_and_preprocess(filepath):
    df = pd.read_excel(filepath)
    return df

def feature_engineering(df):
    def min_max_normalize(series, reverse=False):
        min_a = series.min()
        max_a = series.max()
        if reverse:
            return (series - min_a) / (max_a - min_a)
        else:
            return (max_a - series) / (max_a - min_a)
        # --- I (Intent, 意向深度)：停留时长&浏览页数 各 50% 加权 ---
        df["Time_Spent_Norm"] = min_max_normalize(df["Time_Spent_on_Site_Minutes"])
        df["Pages_Viewed_Norm"] = min_max_normalize(df["Pages_Viewed"])
        df["I_Score"] = 0.5 * df["Time_Spent_Norm"] + 0.5 * df["Pages_Viewed_Norm"]
        # --- Friction (摩擦系数)：浏览多但购买少 => 高摩擦 ---
        df["Friction"] = df["Pages_Viewed"] / (df["Purchase_Frequency"] + 1)

        # --- L (Loyalty, 活跃连接度)：订阅且 7 天内登录=3，仅 7 天内登录=2，其余=1 ---
        def calculate_loyalty(row):
            if row["Newsletter_Subscription"] and row["Last_Login_Days_Ago"] < 7:
                return 3
            elif not row["Newsletter_Subscription"] and row["Last_Login_Days_Ago"] < 7:
                return 2
            return 1

        df["L_Score"] = df.apply(calculate_loyalty, axis=1)
        # --- D (Demographics, 购买力背景)：按收入 33%/66% 分位分 低/中/高 ---
        income_33 = df["Income"].quantile(0.33)
        income_66 = df["Income"].quantile(0.66)

        def income_level(income):
            if income <= income_33:
                return "Low"
            elif income <= income_66:
                return "Medium"
            return "High"
        df["Income_Level"] = df["Income"].apply(income_level)
        # --- 人货匹配度：兴趣 == 偏好类别 ---
        df["Interest_Match"] = (df["Interests"] == df["Product_Category_Preference"]).astype(int)
        print(f"I_Score 范围: [{df['I_Score'].min():.2f}, {df['I_Score'].max():.2f}]")
        print(f"Friction 范围: [{df['Friction'].min():.2f}, {df['Friction'].max():.2f}]")
        print(f"L_Score 分布: {df['L_Score'].value_counts().to_dict()}")
        print(f"Income_Level 分布: {df['Income_Level'].value_counts().to_dict()}")
        print(f"Interest_Match 匹配率: {df['Interest_Match'].mean() * 100:.1f}%")
        return df


