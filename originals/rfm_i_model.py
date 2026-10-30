# -*- coding: utf-8 -*-
"""
电商用户价值分层与精准营销策略 —— 完整版 RFM-I 优化模型（Python 单文件版）

说明：
    本脚本复刻了 ipynb / R 版的完整分析流程，功能等价、可作为独立脚本运行。
    流程共 6 个阶段：
        1. 数据加载与质量检查
        2. 特征工程（I 意向 / Friction 摩擦 / L 忠诚 / 收入分层 / 人货匹配）
        3. EDA 探索性数据分析（分布 / 相关性 / 关键行为洞察）
        4. RFM-I 模型构建（min-max 归一化 + 加权综合分）
        5. 用户分层与画像（8 类基础 + 7 类修正 = 精细化分层 + 雷达图）
        6. ROI 测算（传统 RFM vs 优化 RFI 策略对比）

依赖：pandas, numpy, matplotlib
入口：main()；修改 INPUT_PATH / OUTPUT_DIR 两个常量即可运行。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 0. 常量配置（请按实际路径修改）
# ============================================================
INPUT_PATH = r"D:\新建文件夹1\量化学习\数分项目\user_personalized_features.xlsx"  # 原始 14 列数据，请改成你的实际路径
OUTPUT_DIR = r"D:\新建文件夹1\量化学习\数分项目\result_r"                         # 结果输出目录

# 中文字体设置（若系统无 SimHei 可替换为 "Microsoft YaHei"）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1. 数据加载与质量检查
# ============================================================
def load_and_preprocess(filepath):
    """加载 Excel，打印数据规模 / 缺失值 / 年龄异常值。"""
    print("=" * 60)
    print("第一阶段：数据质量检查")
    print("=" * 60)
    df = pd.read_excel(filepath)
    print(f"数据规模: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"缺失值: {df.isnull().sum().sum()}")
    age_outliers = df[(df["Age"] < 10) | (df["Age"] > 100)].shape[0]
    print(f"年龄异常值: {age_outliers}")
    return df


# ============================================================
# 2. 特征工程：构建 RFM-I 优化指标
# ============================================================
def feature_engineering(df):
    """min-max 归一化辅助函数。"""
    def min_max_normalize(series, reverse=False):
        min_v, max_v = series.min(), series.max()
        if max_v == min_v:                      # 处理全相同列
            return pd.Series([50] * len(series))
        if reverse:                             # 越大越好正向 / 越小越好反向
            return ((max_v - series) / (max_v - min_v)) * 100
        return ((series - min_v) / (max_v - min_v)) * 100

    print("\n" + "=" * 60)
    print("第二阶段：特征工程 - 构建优化指标")
    print("=" * 60)

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


# ============================================================
# 3. EDA 探索性数据分析
# ============================================================
def perform_eda(df, output_dir):
    print("\n" + "=" * 60)
    print("第三阶段：EDA探索性数据分析")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)

    # 3.1 关键分布（4 宫格）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(df["Total_Spending"], bins=50, edgecolor="black",
                    alpha=0.7, color="#0B5CAD")
    axes[0, 0].axvline(df["Total_Spending"].median(), color="red", linestyle="--",
                       label=f'Median: {df["Total_Spending"].median():.0f}')
    axes[0, 0].set_title("Total_Spending Distribution (Long Tail Pattern)")
    axes[0, 0].legend()

    colors = {"Low": "#FF6B6B", "Medium": "#4ECDC4", "High": "#0B5CAD"}
    for level in ["Low", "Medium", "High"]:
        sub = df[df["Income_Level"] == level]
        axes[0, 1].scatter(sub["Income"], sub["Total_Spending"],
                           alpha=0.5, label=level, c=colors[level], s=20)
    axes[0, 1].set_title("Income vs Total_Spending by Income Level")
    axes[0, 1].legend()

    df.boxplot(column="Last_Login_Days_Ago", by="Newsletter_Subscription", ax=axes[1, 0])
    axes[1, 0].set_title("Login Days by Subscription Status")

    axes[1, 1].hist(df["I_Score"], bins=30, edgecolor="black",
                    alpha=0.7, color="#4ECDC4")
    axes[1, 1].set_title("Intent Score (I_Score) Distribution")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eda_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3.2 相关性分析
    numeric_cols = ["Age", "Income", "Last_Login_Days_Ago", "Purchase_Frequency",
                    "Average_Order_Value", "Total_Spending", "Time_Spent_on_Site_Minutes",
                    "Pages_Viewed", "I_Score", "Friction"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols))); ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(numeric_cols, fontsize=9)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    ax.set_title("Feature Correlation Matrix")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3.3 关键行为洞察
    time_pc = df["Time_Spent_on_Site_Minutes"].corr(df["Purchase_Frequency"])
    pages_pc = df["Pages_Viewed"].corr(df["Purchase_Frequency"])
    print(f"\n关键相关性分析:")
    print(f"Time_Spent vs Purchase_Frequency 相关系数: {time_pc:.3f}")
    print(f"Pages_Viewed vs Purchase_Frequency 相关系数: {pages_pc:.3f}")

    high_friction = df[df["Friction"] > df["Friction"].quantile(0.75)]
    print(f"\n高摩擦用户(前25%)特征:")
    print(f"  - 平均浏览页数: {high_friction['Pages_Viewed'].mean():.1f}")
    print(f"  - 平均购买频率: {high_friction['Purchase_Frequency'].mean():.1f}")
    print(f"  - 平均消费金额: {high_friction['Total_Spending'].mean():.0f}")

    high_inc_low_spend = df[(df["Income_Level"] == "High") &
                            (df["Total_Spending"] < df["Total_Spending"].median())]
    print(f"\n高收入低消费群体:")
    print(f"  - 用户数: {len(high_inc_low_spend)}")
    print(f"  - 占高收入群体比例: "
          f"{len(high_inc_low_spend) / len(df[df['Income_Level']=='High']) * 100:.1f}%")


# ============================================================
# 4. RFM-I 模型构建
# ============================================================
def build_rfm_model(df):
    """R/F/M min-max 归一化到 0-100，加权合成 RFM，叠加意向权重得到最终分。"""
    print("\n" + "=" * 60)
    print("第四阶段：RFM-I模型构建")
    print("=" * 60)
    def min_max_normalize(series, reverse=False):
        min_v, max_v = series.min(), series.max()
        if max_v == min_v:
            return pd.Series([50] * len(series))
        if reverse:
            return ((max_v - series) / (max_v - min_v)) * 100
        return ((series - min_v) / (max_v - min_v)) * 100

    df["R_Score"] = min_max_normalize(df["Last_Login_Days_Ago"], reverse=True)  # 登录越近分越高
    df["F_Score"] = min_max_normalize(df["Purchase_Frequency"])                 # 频次越高分越高
    df["M_Score"] = min_max_normalize(df["Total_Spending"])                     # 金额越高分越高
    df["RFM_Score"] = 0.2 * df["R_Score"] + 0.3 * df["F_Score"] + 0.5 * df["M_Score"]
    df["I_Weight"] = df["I_Score"] / 500            # I_Score 0-100 -> 0-0.2 权重
    df["Final_Score"] = df["RFM_Score"] * (1 + df["I_Weight"])

    for c in ["R_Score", "F_Score", "M_Score", "RFM_Score", "Final_Score"]:
        print(f"{c} 范围: [{df[c].min():.2f}, {df[c].max():.2f}]")
    return df


# ============================================================
# 5. 用户分层与画像
# ============================================================
def classify_users(df):
    """8 类基础 RFM 分层 + 7 类特殊人群修正 => 精细化分层。"""
    print("\n" + "=" * 60)
    print("第五阶段：用户分层与画像")
    print("=" * 60)
    friction_q60 = df["Friction"].quantile(0.6)

    def classify_user(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        i, income, friction, l = (row["I_Score"], row["Income_Level"],
                                  row["Friction"], row["L_Score"])

        # --- 基础 RFM 八类 ---
        if r > 60 and f > 60 and m > 60:
            base = "重要价值用户"
        elif r > 60 and m > 60 and f < 40:
            base = "重要发展用户"
        elif r < 40 and f > 60 and m > 60:
            base = "重要保持用户"
        elif r < 40 and m > 60:
            base = "重要挽留用户"
        elif r > 60 and f > 40 and m < 40:
            base = "一般发展用户"
        elif r > 60 and f < 40 and m < 40:
            base = "一般维持用户"
        elif r < 40 and f < 40 and m < 40:
            base = "低价值用户"
        else:
            base = "一般用户"

        # --- 关键修正逻辑：识别特殊人群 ---
        if base in ("低价值用户", "一般维持用户") and income == "High" and i > 60:
            return "高潜沉睡用户"
        if income == "High" and i > 70 and f < 40 and m < 50:
            return "纠结土豪"
        if base in ("一般维持用户", "一般发展用户") and i > 80 and friction > friction_q60:
            return "犹豫型潜力用户"
        if income == "Low" and i > 70 and f < 40:
            return "隐形活跃者"
        if income == "High" and r < 40 and m > 50:
            return "高潜流失客"
        if m > 70 and f > 70 and i > 60:
            return "核心VIP"
        if income == "Low" and i < 40 and m < 40:
            return "羊毛党/低值"
        return base

    df["User_Segment"] = df.apply(classify_user, axis=1)

    stats = df.groupby("User_Segment").agg({
        "User_ID": "count",
        "Total_Spending": "mean",
        "Purchase_Frequency": "mean",
        "I_Score": "mean",
        "Income": "mean",
        "Final_Score": "mean",
    }).round(2)
    stats.columns = ["用户数", "平均消费", "平均购买频率", "平均意向分", "平均收入", "平均综合分"]
    stats["占比(%)"] = (stats["用户数"] / len(df) * 100).round(1)
    stats = stats.sort_values("用户数", ascending=False)
    print("\n用户分层统计:")
    print(stats.to_string())
    return df, stats


# ============================================================
# 6. 用户画像可视化（雷达图 + 分布柱状图）
# ============================================================
def generate_portrait(df, output_dir):
    segments = df["User_Segment"].unique()
    metrics = ["R_Score", "F_Score", "M_Score", "I_Score"]
    means = df.groupby("User_Segment")[metrics].mean()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw=dict(polar=True))
    axes = axes.flatten()
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))

    for idx, seg in enumerate(segments[:8]):
        ax = axes[idx]
        values = means.loc[seg].tolist() + [means.loc[seg].iloc[0]]
        ax.fill(angles, values, alpha=0.3, color=colors[idx])
        ax.plot(angles, values, "o-", color=colors[idx], linewidth=2)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(["R", "F", "M", "I"], fontsize=10)
        ax.set_title(seg, fontsize=10, pad=10)
    for idx in range(min(len(segments), 8), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/segment_radar.png", dpi=150, bbox_inches="tight")
    plt.close()

    counts = df["User_Segment"].value_counts()
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(counts.index, counts.values, color="#0B5CAD", alpha=0.8)
    ax.set_xlabel("User Count"); ax.set_title("User Segment Distribution")
    for bar, cnt in zip(bars, counts.values):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"{cnt} ({cnt/len(df)*100:.1f}%)", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/segment_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 7. ROI 测算：传统 RFM vs 优化 RFI
# ============================================================
def calculate_roi(df, output_dir):
    print("\n" + "=" * 60)
    print("第六阶段：ROI测算")
    print("=" * 60)
    total_budget = 10000            # 总预算（元）
    coupon_cost = 10                # 每张优惠券成本（元）
    aov = df["Average_Order_Value"].mean()

    # 转化率假设（可用历史数据校准）
    core_n = 0.25;  core_w = 0.30                      # 核心用户 自然/发券
    pot_n = 0.01;   pot_w = 0.20                       # 潜力用户 自然/发券
    rfm20_n = 0.20; rfm20_w = 0.30                     # RFM 前20% 自然/发券
    core_lift = core_w - core_n
    pot_lift = pot_w - pot_n
    rfm20_lift = rfm20_w - rfm20_n

    # ---- 方案A：传统 RFM 前20% ----
    top20 = df["RFM_Score"].quantile(0.80)
    rfm_top20 = df[df["RFM_Score"] >= top20]
    cnt_a = min(len(rfm_top20), total_budget // coupon_cost)
    cost_a = cnt_a * coupon_cost
    rev_a = cnt_a * rfm20_lift * aov
    roi_a = (rev_a - cost_a) / cost_a * 100 if cost_a > 0 else 0

    # ---- 方案B：优化 RFI 分层 ----
    core = df[df["User_Segment"].isin(["核心VIP", "重要价值用户"])]
    pot = df[df["User_Segment"].isin(["纠结土豪", "高潜沉睡用户", "犹豫型潜力用户", "高潜流失客"])]
    core_count = min(len(core), int(total_budget * 0.8 / coupon_cost))
    remaining = total_budget - core_count * coupon_cost
    pot_count = min(len(pot), remaining // coupon_cost)
    cost_b = core_count * coupon_cost + pot_count * coupon_cost
    rev_b = core_count * core_lift * aov + pot_count * pot_lift * aov
    roi_b = (rev_b - cost_b) / cost_b * 100 if cost_b > 0 else 0

    print(f"  总预算: {total_budget}元 | 优惠券成本: {coupon_cost}元/人 | 平均客单价: {aov:.0f}元")
    print(f"\n【方案A - 传统RFM策略】")
    print(f"  目标用户数: {cnt_a} | 成本: {cost_a}元 | 边际收益: {rev_a:.0f}元 | 边际ROI: {roi_a:.1f}%")
    print(f"\n【方案B - 优化RFI策略】")
    print(f"  核心用户: {core_count} | 潜力用户: {pot_count} | 目标总数: {core_count+pot_count}")
    print(f"  成本: {cost_b}元 (预算使用率: {cost_b/total_budget*100:.1f}%)")
    print(f"  边际收益: {rev_b:.0f}元 | 边际ROI: {roi_b:.1f}%")

    # ROI 对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    strategies = ["Traditional RFM", "Optimized RFI"]
    rois = [roi_a, roi_b]
    bars = axes[0].bar(strategies, rois, color=["#FF6B6B", "#0B5CAD"], alpha=0.8,
                       edgecolor="black")
    for bar, roi in zip(bars, rois):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"{roi:.1f}%", ha="center", fontweight="bold")
    axes[0].set_ylabel("边际ROI (%)"); axes[0].set_title("边际ROI对比：传统RFM vs 优化RFI")

    x = np.arange(2); width = 0.35
    core_marg = [rev_a, core_count * core_lift * aov]
    pot_marg = [0, pot_count * pot_lift * aov]
    axes[1].bar(x, core_marg, width, label="核心用户边际收益", color="#0B5CAD", alpha=0.8)
    axes[1].bar(x, pot_marg, width, bottom=core_marg, label="潜力用户边际收益",
                color="#4ECDC4", alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(strategies)
    axes[1].set_ylabel("边际收益 (元)"); axes[1].set_title("边际收益构成对比")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roi_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {"strategy_a": {"count": cnt_a, "cost": cost_a, "rev": rev_a, "roi": roi_a},
            "strategy_b": {"core": core_count, "pot": pot_count, "cost": cost_b,
                           "rev": rev_b, "roi": roi_b}}


# ============================================================
# 主流程
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_and_preprocess(INPUT_PATH)
    df = feature_engineering(df)
    perform_eda(df, OUTPUT_DIR)
    df = build_rfm_model(df)
    df, stats = classify_users(df)
    generate_portrait(df, OUTPUT_DIR)
    roi = calculate_roi(df, OUTPUT_DIR)

    # 输出完整结果
    df.to_excel(os.path.join(OUTPUT_DIR, "rfm_analysis_full.xlsx"), index=False)
    stats.to_excel(os.path.join(OUTPUT_DIR, "segment_stats.xlsx"))
    print("\n全部完成，结果已保存到:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
