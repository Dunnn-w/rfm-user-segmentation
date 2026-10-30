# -*- coding: utf-8 -*-
"""
电商用户价值分层与精准营销策略 —— 优化版主程序
================================================

在原始 rfm_i_model.py / 经典rfm/rfm_model.py 基础上的整合优化工程：

  * 统一两条分析链路于一个入口：
       经典 RFM   : 分位数 1-5 打分 + 等权合成 + 中位数 8 类分层
       RFM-I 优化 : min-max 归一化 0-100 + 加权合成(0.2R+0.3F+0.5M)
                    + 意向加成 + 8类基础 + 7类修正 = 精细化分层
  * 修复早期草稿 rfm_i_models.py 中 min_max_normalize 方向写反的 bug；
    归一化函数全局唯一，消除两版脚本重复定义、口径不一致的问题。
  * 所有可调参数（权重/阈值/预算/转化率/字体）集中在 Config，便于调参。
  * 路径自动解析（相对本文件），拷贝到任意目录即可运行；输出统一进 output/。
  * 中文字体自动回退：SimHei -> Microsoft YaHei。

用法：
    python main.py            # 完整流程（经典 RFM + RFM-I 对比）
    python main.py classic    # 仅经典 RFM 链路
    python main.py rfmi       # 仅 RFM-I 优化链路

依赖：pandas numpy matplotlib openpyxl（见 requirements.txt）
"""

import os
import sys
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 0. 配置（集中管理所有可调参数）
# ============================================================
@dataclass
class Config:
    # ---- 路径（默认相对本文件，可自行覆盖） ----
    data_file: str = os.path.join(BASE_DIR, "data", "user_personalized_features.xlsx")
    output_dir: str = os.path.join(BASE_DIR, "output")

    # ---- 中文字体（自动选用第一个可用的） ----
    fonts: list = field(default_factory=lambda: ["SimHei", "Microsoft YaHei", "SimSun"])

    # ---- RFM-I 打分权重（0.2R + 0.3F + 0.5M） ----
    w_r: float = 0.2
    w_f: float = 0.3
    w_m: float = 0.5
    i_weight_denom: float = 500.0        # I_Score / 500 -> 意向加成上限 0.2（即 +20%）

    # ---- 分层阈值（R/F/M 0-100 分） ----
    high_th: float = 60.0                # >60 视为"高"
    low_th: float = 40.0                 # <40 视为"低"
    friction_quantile: float = 0.6       # 犹豫型潜力用户的摩擦阈值分位

    # ---- 修正类识别参数 ----
    adj_high_income: float = 60.0        # 高潜沉睡用户：I 阈值
    adj_rich_confused: float = 70.0      # 纠结土豪：I 阈值
    adj_hidden_active: float = 70.0      # 隐形活跃者：I 阈值
    adj_vip_i: float = 60.0              # 核心VIP：I 阈值
    adj_vip_m: float = 70.0              # 核心VIP：M 阈值
    adj_vip_f: float = 70.0              # 核心VIP：F 阈值
    adj_lost_m: float = 50.0             # 高潜流失客：M 阈值
    adj_wool_i: float = 40.0             # 羊毛党：I 阈值（低于）
    adj_wool_m: float = 40.0             # 羊毛党：M 阈值（低于）

    # ---- ROI 测算参数 ----
    total_budget: float = 10000.0        # 总预算（元）
    coupon_cost: float = 10.0            # 每张优惠券成本（元）
    conv_core: tuple = (0.25, 0.30)      # 核心用户（自然转化率, 发券后转化率）
    conv_pot: tuple = (0.01, 0.20)       # 潜力用户
    conv_rfm20: tuple = (0.20, 0.30)     # RFM 前20%
    plan_a_quantile: float = 0.80        # 方案A：RFM_Score 前20%分位
    plan_b_core_ratio: float = 0.80      # 方案B：核心用户预算占比


CFG = Config()


# ============================================================
# 0.1 工具函数
# ============================================================
def banner(text: str) -> None:
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)


def setup_chinese_font() -> None:
    """设置中文字体（按可用性自动回退）。"""
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    pick = next((f for f in CFG.fonts if f in available), None)
    if pick:
        plt.rcParams["font.sans-serif"] = [pick]
        print(f"[字体] 使用中文字体: {pick}")
    else:
        print("[字体] 警告：未找到 SimHei/Microsoft YaHei，中文可能显示为方块")
    plt.rcParams["axes.unicode_minus"] = False


def min_max_normalize(series: pd.Series, reverse: bool = False) -> pd.Series:
    """
    归一化到 0-100。
    reverse=False：数值越大分越高（正向指标，如 消费金额/购买频次）
    reverse=True ：数值越小分越高（反向指标，如 距上次登录天数）
    特例：列内全相同 -> 统一给 50（中性分）。
    """
    min_v, max_v = float(series.min()), float(series.max())
    if max_v == min_v:
        return pd.Series(50.0, index=series.index)
    if reverse:
        return (max_v - series) / (max_v - min_v) * 100.0
    return (series - min_v) / (max_v - min_v) * 100.0


# ============================================================
# 1. 数据加载与质量检查
# ============================================================
def load_and_quality_check(filepath: str) -> pd.DataFrame:
    banner("第一步：数据加载与质量检查")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"输入文件不存在: {filepath}\n"
            f"请先用 extract_raw_data.py 重建用户数据，或修改 Config.data_file。"
        )
    df = pd.read_excel(filepath)
    n_row, n_col = df.shape
    print(f"数据规模: {n_row} 行 x {n_col} 列")
    print(f"缺失值总数: {int(df.isnull().sum().sum())}")
    print(f"重复 User_ID: {int(df['User_ID'].duplicated().sum())}")

    if "Age" in df.columns:
        age_bad = int(((df["Age"] < 10) | (df["Age"] > 100)).sum())
        print(f"年龄异常值(<10 或 >100): {age_bad}")
        if age_bad > 0:
            print("  -> 提示：按题意此类样本应剔除或复核；本例为演示数据，暂不清洗（与原始逻辑一致）。")
    return df


# ============================================================
# 2. 特征工程：构建 RFM-I 优化指标
# ============================================================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    banner("第二步：特征工程（意向度 / 摩擦系数 / 忠诚度 / 收入分层 / 人货匹配）")

    # --- I (Intent, 意向深度)：停留时长 & 浏览页数 各 50% 加权 ---
    df["Time_Spent_Norm"] = min_max_normalize(df["Time_Spent_on_Site_Minutes"])
    df["Pages_Viewed_Norm"] = min_max_normalize(df["Pages_Viewed"])
    df["I_Score"] = 0.5 * df["Time_Spent_Norm"] + 0.5 * df["Pages_Viewed_Norm"]

    # --- Friction (摩擦系数)：浏览多但购买少 => 高摩擦（仅用于人群洞察与修正分层） ---
    df["Friction"] = df["Pages_Viewed"] / (df["Purchase_Frequency"] + 1)

    # --- L (Loyalty, 活跃连接度)：订阅且7天内登录=3；仅7天内登录=2；其余=1（向量化） ---
    sub = df["Newsletter_Subscription"].astype(bool)
    recent = df["Last_Login_Days_Ago"] < 7
    df["L_Score"] = np.where(sub & recent, 3, np.where(recent, 2, 1))

    # --- D (购买力背景)：按收入 33%/66% 分位切 低/中/高 ---
    income_33, income_66 = df["Income"].quantile([0.33, 0.66])
    df["Income_Level"] = df["Income"].apply(
        lambda x: "Low" if x <= income_33 else ("Medium" if x <= income_66 else "High")
    )

    # --- 人货匹配度：兴趣 == 偏好品类 ---
    df["Interest_Match"] = (df["Interests"] == df["Product_Category_Preference"]).astype(int)

    print(f"I_Score 范围: [{df['I_Score'].min():.2f}, {df['I_Score'].max():.2f}]")
    print(f"Friction 范围: [{df['Friction'].min():.2f}, {df['Friction'].max():.2f}]")
    print(f"L_Score 分布: {df['L_Score'].value_counts().sort_index().to_dict()}")
    print(f"Income_Level 分布: {df['Income_Level'].value_counts().to_dict()}")
    print(f"Interest_Match 匹配率: {df['Interest_Match'].mean() * 100:.1f}%")
    return df


# ============================================================
# 3. EDA 探索性数据分析
# ============================================================
def perform_eda(df: pd.DataFrame, out_dir: str) -> None:
    banner("第三步：EDA 探索性数据分析")
    numeric_cols = ["Age", "Income", "Last_Login_Days_Ago", "Purchase_Frequency",
                    "Average_Order_Value", "Total_Spending", "Time_Spent_on_Site_Minutes",
                    "Pages_Viewed", "I_Score", "Friction"]

    # 3.1 关键分布（4 宫格）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(df["Total_Spending"], bins=50, edgecolor="black", alpha=0.7, color="#0B5CAD")
    med = df["Total_Spending"].median()
    axes[0, 0].axvline(med, color="red", linestyle="--", label=f"Median: {med:.0f}")
    axes[0, 0].set_title("Total_Spending 分布（长尾）")
    axes[0, 0].legend()

    colors = {"Low": "#FF6B6B", "Medium": "#4ECDC4", "High": "#0B5CAD"}
    for level in ["Low", "Medium", "High"]:
        sub = df[df["Income_Level"] == level]
        axes[0, 1].scatter(sub["Income"], sub["Total_Spending"], alpha=0.5,
                           label=level, c=colors[level], s=20)
    axes[0, 1].set_title("收入 vs 消费（按收入层）")
    axes[0, 1].legend()

    df.boxplot(column="Last_Login_Days_Ago", by="Newsletter_Subscription", ax=axes[1, 0])
    axes[1, 0].set_title("登录间隔 vs 订阅状态")

    axes[1, 1].hist(df["I_Score"], bins=30, edgecolor="black", alpha=0.7, color="#4ECDC4")
    axes[1, 1].set_title("意向度 I_Score 分布")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3.2 相关性热力图
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(numeric_cols, fontsize=9)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    ax.set_title("特征相关性矩阵")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3.3 关键行为洞察
    print("关键相关性：")
    print(f"  Time_Spent on Site vs Purchase_Frequency : {df['Time_Spent_on_Site_Minutes'].corr(df['Purchase_Frequency']):.3f}")
    print(f"  Pages_Viewed        vs Purchase_Frequency : {df['Pages_Viewed'].corr(df['Purchase_Frequency']):.3f}")
    hf = df[df["Friction"] > df["Friction"].quantile(0.75)]
    print(f"高摩擦用户(前25%)：浏览页数均值 {hf['Pages_Viewed'].mean():.1f} / 购买频次均值 {hf['Purchase_Frequency'].mean():.1f} / 消费均值 {hf['Total_Spending'].mean():.0f}")
    hils = df[(df["Income_Level"] == "High") & (df["Total_Spending"] < df["Total_Spending"].median())]
    print(f"高收入低消费群体：{len(hils)} 人，占高收入群体 {len(hils) / max(len(df[df['Income_Level']=='High']), 1) * 100:.1f}%")


# ============================================================
# 4. 模型 A：经典 RFM（分位数 1-5 打分 + 等权合成 + 中位数 8 类）
# ============================================================
def classic_rfm(df: pd.DataFrame) -> pd.DataFrame:
    banner("第四步A：经典 RFM 模型（1-5 分位打分 → 0-100 → 等权合成）")
    rfm = df[["User_ID", "Last_Login_Days_Ago", "Purchase_Frequency", "Total_Spending"]].copy()

    # 为避免 qcut 因重复值报错，统一先做 rank（monotone 保序），再 qcut 5 组
    rfm["c_R_raw"] = rfm["Last_Login_Days_Ago"]
    rfm["c_F_raw"] = rfm["Purchase_Frequency"]
    rfm["c_M_raw"] = rfm["Total_Spending"]

    # R：天数越小越好 -> 高分在前；F/M：越大越好 -> 高分在后
    rfm["c_R_score"] = pd.qcut(rfm["c_R_raw"].rank(method="first"), q=5,
                               labels=[5, 4, 3, 2, 1]).astype(float)
    rfm["c_F_score"] = pd.qcut(rfm["c_F_raw"].rank(method="first"), q=5,
                               labels=[1, 2, 3, 4, 5]).astype(float)
    rfm["c_M_score"] = pd.qcut(rfm["c_M_raw"].rank(method="first"), q=5,
                               labels=[1, 2, 3, 4, 5]).astype(float)

    # 1-5 分 -> 0-100
    for c in ["c_R_score", "c_F_score", "c_M_score"]:
        rfm[c] = (rfm[c] - 1) / 4 * 100

    # 经典 RFM 综合分（等权）
    rfm["c_RFM_Score"] = (rfm["c_R_score"] + rfm["c_F_score"] + rfm["c_M_score"]) / 3

    # 中位数切 高/低 -> 8 类
    r_med, f_med, m_med = (rfm["c_R_score"].median(), rfm["c_F_score"].median(),
                           rfm["c_M_score"].median())
    rfm["c_R_level"] = np.where(rfm["c_R_score"] >= r_med, "高", "低")
    rfm["c_F_level"] = np.where(rfm["c_F_score"] >= f_med, "高", "低")
    rfm["c_M_level"] = np.where(rfm["c_M_score"] >= m_med, "高", "低")
    segment_map = {
        ("高", "高", "高"): "重要价值客户", ("高", "高", "低"): "重要发展客户",
        ("高", "低", "高"): "重要保持客户", ("高", "低", "低"): "重要挽留客户",
        ("低", "高", "高"): "一般价值客户", ("低", "高", "低"): "一般发展客户",
        ("低", "低", "高"): "一般保持客户", ("低", "低", "低"): "一般挽留客户",
    }
    rfm["Classic_Segment"] = [
        segment_map[(r, f_, m)] for r, f_, m in zip(rfm["c_R_level"], rfm["c_F_level"], rfm["c_M_level"])
    ]

    keep = ["User_ID", "c_R_score", "c_F_score", "c_M_score", "c_RFM_Score", "Classic_Segment"]
    out = df.merge(rfm[keep], on="User_ID", how="left")
    print("经典 RFM 分层汇总：")
    print(out["Classic_Segment"].value_counts().to_string())
    return out


# ============================================================
# 5. 模型 B：RFM-I 优化模型（min-max 加权 + 意向加成）
# ============================================================
def build_rfmi_scores(df: pd.DataFrame) -> pd.DataFrame:
    banner("第四步B：RFM-I 模型打分（min-max 0-100 + 加权合成 + 意向加成）")
    df["R_Score"] = min_max_normalize(df["Last_Login_Days_Ago"], reverse=True)   # 登录越近分越高
    df["F_Score"] = min_max_normalize(df["Purchase_Frequency"])                  # 频次越高分越高
    df["M_Score"] = min_max_normalize(df["Total_Spending"])                      # 金额越高分越高
    df["RFM_Score"] = (CFG.w_r * df["R_Score"] + CFG.w_f * df["F_Score"] + CFG.w_m * df["M_Score"])
    df["I_Weight"] = df["I_Score"] / CFG.i_weight_denom                          # 0-100 -> 0-0.2
    df["Final_Score"] = df["RFM_Score"] * (1 + df["I_Weight"])                   # 最高 120
    for c in ["R_Score", "F_Score", "M_Score", "RFM_Score", "Final_Score"]:
        print(f"  {c:12s} 范围: [{df[c].min():6.2f}, {df[c].max():6.2f}]")
    return df


# ============================================================
# 6. RFM-I 用户分层（8 类基础 + 7 类修正）
# ============================================================
def classify_rfmi(df: pd.DataFrame) -> pd.DataFrame:
    banner("第五步：RFM-I 用户分层（8 类基础 + 7 类修正 = 精细化分层）")
    friction_q60 = df["Friction"].quantile(CFG.friction_quantile)

    def classify_user(row) -> str:
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        i = row["I_Score"]
        income, friction = row["Income_Level"], row["Friction"]

        # --- 基础 RFM 八类 ---
        if r > CFG.high_th and f > CFG.high_th and m > CFG.high_th:
            base = "重要价值用户"
        elif r > CFG.high_th and m > CFG.high_th and f < CFG.low_th:
            base = "重要发展用户"
        elif r < CFG.low_th and f > CFG.high_th and m > CFG.high_th:
            base = "重要保持用户"
        elif r < CFG.low_th and m > CFG.high_th:
            base = "重要挽留用户"
        elif r > CFG.high_th and f > CFG.low_th and m < CFG.low_th:
            base = "一般发展用户"
        elif r > CFG.high_th and f < CFG.low_th and m < CFG.low_th:
            base = "一般维持用户"
        elif r < CFG.low_th and f < CFG.low_th and m < CFG.low_th:
            base = "低价值用户"
        else:
            base = "一般用户"

        # --- 修正逻辑：识别特殊人群 ---
        if base in ("低价值用户", "一般维持用户") and income == "High" and i > CFG.adj_high_income:
            return "高潜沉睡用户"
        if income == "High" and i > CFG.adj_rich_confused and f < CFG.low_th and m < 50:
            return "纠结土豪"
        if base in ("一般维持用户", "一般发展用户") and i > 80 and friction > friction_q60:
            return "犹豫型潜力用户"
        if income == "Low" and i > CFG.adj_hidden_active and f < CFG.low_th:
            return "隐形活跃者"
        if income == "High" and r < CFG.low_th and m > CFG.adj_lost_m:
            return "高潜流失客"
        if m > CFG.adj_vip_m and f > CFG.adj_vip_f and i > CFG.adj_vip_i:
            return "核心VIP"
        if income == "Low" and i < CFG.adj_wool_i and m < CFG.adj_wool_m:
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
        "R_Score": "mean",
        "F_Score": "mean",
        "M_Score": "mean",
    }).round(2)
    stats.columns = ["用户数", "平均消费", "平均购买频率", "平均意向分", "平均收入",
                     "平均综合分", "平均R", "平均F", "平均M"]
    stats["占比(%)"] = (stats["用户数"] / len(df) * 100).round(1)
    stats = stats.sort_values("用户数", ascending=False)
    print(stats.to_string())
    return df


# ============================================================
# 7. 用户画像可视化（雷达图 + 分布柱状图）
# ============================================================
def generate_portrait(df: pd.DataFrame, out_dir: str) -> None:
    banner("第六步：用户画像可视化")
    metrics = ["R_Score", "F_Score", "M_Score", "I_Score"]
    counts = df["User_Segment"].value_counts()
    segments = counts.index.tolist()
    means = df.groupby("User_Segment")[metrics].mean()

    # 雷达图：按人数取前 8 类（比原版按出现顺序取更合理）
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw=dict(polar=True))
    axes = axes.flatten()
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
    top = segments[:8]

    for idx, seg in enumerate(top):
        ax = axes[idx]
        base = means.loc[seg]
        values = base.tolist() + [base.iloc[0]]
        ax.fill(angles, values, alpha=0.3, color=colors[idx])
        ax.plot(angles, values, "o-", color=colors[idx], linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(["R", "F", "M", "I"], fontsize=10)
        ax.set_title(f"{seg} ({counts[seg]})", fontsize=10, pad=10)
    for idx in range(len(top), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "segment_radar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 分布柱状图（全部分层）
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(segments, counts.values, color="#0B5CAD", alpha=0.8)
    ax.set_xlabel("用户数")
    ax.set_title("RFM-I 分层用户分布")
    for bar, cnt in zip(bars, counts.values):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"{cnt} ({cnt / len(df) * 100:.1f}%)", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "segment_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"共识别 {len(segments)} 个细分人群，雷达图展示人数前 {len(top)} 类。")


# ============================================================
# 8. ROI 测算：传统 RFM vs 优化 RFI
# ============================================================
def calculate_roi(df: pd.DataFrame, out_dir: str) -> dict:
    banner("第七步：ROI 测算（方案A 传统RFM vs 方案B 优化RFI）")
    aov = df["Average_Order_Value"].mean()
    core_n, core_w = CFG.conv_core
    pot_n, pot_w = CFG.conv_pot
    rfm20_n, rfm20_w = CFG.conv_rfm20
    core_lift, pot_lift, rfm20_lift = core_w - core_n, pot_w - pot_n, rfm20_w - rfm20_n

    # ---- 方案A：传统 RFM 前 20% ----
    top_q = df["RFM_Score"].quantile(CFG.plan_a_quantile)
    rfm_top20 = df[df["RFM_Score"] >= top_q]
    cnt_a = min(len(rfm_top20), int(CFG.total_budget // CFG.coupon_cost))
    cost_a = cnt_a * CFG.coupon_cost
    rev_a = cnt_a * rfm20_lift * aov
    roi_a = (rev_a - cost_a) / cost_a * 100 if cost_a > 0 else 0.0

    # ---- 方案B：优化 RFI 分层（核心 80% 预算 + 潜力 20% 预算） ----
    core = df[df["User_Segment"].isin(["核心VIP", "重要价值用户"])]
    pot = df[df["User_Segment"].isin(["纠结土豪", "高潜沉睡用户", "犹豫型潜力用户", "高潜流失客"])]
    core_count = min(len(core), int(CFG.total_budget * CFG.plan_b_core_ratio / CFG.coupon_cost))
    remaining = CFG.total_budget - core_count * CFG.coupon_cost
    pot_count = min(len(pot), int(remaining // CFG.coupon_cost))
    cost_b = core_count * CFG.coupon_cost + pot_count * CFG.coupon_cost
    rev_b = core_count * core_lift * aov + pot_count * pot_lift * aov
    roi_b = (rev_b - cost_b) / cost_b * 100 if cost_b > 0 else 0.0

    print(f"总预算 {CFG.total_budget:.0f} 元 | 券成本 {CFG.coupon_cost:.0f} 元/人 | 平均客单价 {aov:.0f} 元")
    print(f"转化率假设：核心 {core_n:.0%}->{core_w:.0%} / 潜力 {pot_n:.0%}->{pot_w:.0%} / RFM前20% {rfm20_n:.0%}->{rfm20_w:.0%}")
    print(f"[方案A] 目标 {cnt_a} 人 | 成本 {cost_a:.0f} 元 | 边际收益 {rev_a:.0f} 元 | 边际ROI {roi_a:.1f}%")
    print(f"[方案B] 核心 {core_count} + 潜力 {pot_count} = {core_count + pot_count} 人 | "
          f"成本 {cost_b:.0f} 元 (预算使用率 {cost_b / CFG.total_budget * 100:.1f}%) | "
          f"边际收益 {rev_b:.0f} 元 | 边际ROI {roi_b:.1f}%")

    # ROI 对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    strategies = ["传统 RFM", "优化 RFI"]
    rois = [roi_a, roi_b]
    bars = axes[0].bar(strategies, rois, color=["#FF6B6B", "#0B5CAD"], alpha=0.8, edgecolor="black")
    for bar, roi in zip(bars, rois):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"{roi:.1f}%", ha="center", fontweight="bold")
    axes[0].set_ylabel("边际ROI (%)")
    axes[0].set_title("边际ROI对比：传统 RFM vs 优化 RFI")

    x = np.arange(2)
    width = 0.35
    core_marg = [rev_a, core_count * core_lift * aov]
    pot_marg = [0, pot_count * pot_lift * aov]
    axes[1].bar(x, core_marg, width, label="核心用户边际收益", color="#0B5CAD", alpha=0.8)
    axes[1].bar(x, pot_marg, width, bottom=core_marg, label="潜力用户边际收益", color="#4ECDC4", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(strategies)
    axes[1].set_ylabel("边际收益 (元)")
    axes[1].set_title("边际收益构成对比")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roi_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "方案A_传统RFM": {"目标用户数": cnt_a, "成本": round(cost_a), "边际收益": round(rev_a), "边际ROI%": round(roi_a, 1)},
        "方案B_优化RFI": {"目标用户数": core_count + pot_count, "成本": round(cost_b),
                         "边际收益": round(rev_b), "边际ROI%": round(roi_b, 1)},
    }


# ============================================================
# 9. 输出（Excel + 汇总）
# ============================================================
def write_outputs(df: pd.DataFrame, roi, out_dir: str) -> None:
    banner("第八步：导出结果")
    os.makedirs(out_dir, exist_ok=True)

    full = df.copy()
    full.to_excel(os.path.join(out_dir, "rfm_analysis_full.xlsx"), index=False)

    # RFM-I 分层统计（仅 RFM-I 链路存在时）
    if "User_Segment" in df.columns:
        stats = df.groupby("User_Segment").agg({
            "User_ID": "count", "R_Score": "mean", "F_Score": "mean", "M_Score": "mean",
            "RFM_Score": "mean", "I_Score": "mean", "Final_Score": "mean",
            "Total_Spending": "mean", "Income": "mean",
        }).round(2)
        stats.columns = ["用户数", "平均R", "平均F", "平均M", "平均RFM", "平均意向分",
                         "平均综合分", "平均消费", "平均收入"]
        stats["占比(%)"] = (stats["用户数"] / len(df) * 100).round(2)
        stats = stats.sort_values("用户数", ascending=False)
        stats.to_excel(os.path.join(out_dir, "segment_stats.xlsx"))

    # 经典 RFM 明细（仅经典链路存在时）
    classic_cols = ["User_ID", "c_R_score", "c_F_score", "c_M_score", "c_RFM_Score", "Classic_Segment"]
    if all(c in df.columns for c in classic_cols):
        df[classic_cols].to_excel(os.path.join(out_dir, "classic_rfm_result.xlsx"), index=False)

    if roi is not None:
        pd.DataFrame(roi).T.to_excel(os.path.join(out_dir, "roi_summary.xlsx"))

    print(f"结果已输出到: {out_dir}")
    print("  - rfm_analysis_full.xlsx   （全量明细：原始+衍生+双模型分数+分层）")
    if "User_Segment" in df.columns:
        print("  - segment_stats.xlsx       （RFM-I 分层统计）")
    if all(c in df.columns for c in classic_cols):
        print("  - classic_rfm_result.xlsx  （经典 RFM 分数与分层）")
    if roi is not None:
        print("  - roi_summary.xlsx         （ROI 两方案对比）")
    print("  - *.png                    （EDA/画像/ROI 图）")


# ============================================================
# 主流程
# ============================================================
def main(mode: str = "all") -> None:
    t0 = time.time()
    setup_chinese_font()
    os.makedirs(CFG.output_dir, exist_ok=True)

    df = load_and_quality_check(CFG.data_file)
    df = feature_engineering(df)
    perform_eda(df, CFG.output_dir)

    if mode in ("classic", "all"):
        df = classic_rfm(df)

    if mode in ("rfmi", "all"):
        df = build_rfmi_scores(df)
        df = classify_rfmi(df)
        generate_portrait(df, CFG.output_dir)

    # ROI 测算依赖 RFM-I 打分列；仅经典链路时跳过（与原始经典脚本行为一致）
    roi = calculate_roi(df, CFG.output_dir) if "RFM_Score" in df.columns else None
    write_outputs(df, roi, CFG.output_dir)

    print(f"\n[完成] 全流程用时 {time.time() - t0:.1f}s（模式: {mode}）")


if __name__ == "__main__":
    mode_arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if mode_arg not in ("classic", "rfmi", "all"):
        print(f"未知模式: {mode_arg}，可选 classic / rfmi / all")
        sys.exit(1)
    main(mode_arg)