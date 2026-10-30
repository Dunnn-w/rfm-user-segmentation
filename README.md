# 电商用户价值分层与精准营销策略 —— 优化版工程

在原始 `rfm_i_model.py`（RFM-I 优化模型）与 `经典rfm/rfm_model.py`（经典 RFM）
基础上整合、重构的**单入口工程**：一个入口同时跑通两条模型链路并做 ROI 对比。

## 一、工程结构

```
电商用户价值分层_RFM-I_优化版/
├── main.py                 ★ 优化版主程序（唯一入口）
├── extract_raw_data.py        数据重建工具（见"数据说明"）
├── requirements.txt           依赖清单
├── README.md                  本说明
├── data/
│   └── user_personalized_features.xlsx   原始 14 列用户数据（已重建）
└── output/                     运行后自动生成（图表 + Excel 结果）
```

## 二、快速开始

```bash
pip install -r requirements.txt
python main.py            # 完整流程（经典 RFM + RFM-I 对比）
python main.py classic    # 仅经典 RFM 链路
python main.py rfmi       # 仅 RFM-I 优化链路
```

> 路径自动解析：所有输入输出均相对本文件所在目录，把整个文件夹拷贝到
> 任意位置即可直接运行，无需改路径常量。
> 中文字体自动回退（SimHei -> Microsoft YaHei），无需手动配置。

## 三、两条模型链路（核心差异）

| 维度        | 经典 RFM                                | RFM-I 优化模型                                   |
|-------------|-----------------------------------------|--------------------------------------------------|
| R 打分      | 分位数 1-5（登录越近分越高）            | min-max 归一化 0-100（反向）                      |
| F/M 打分    | 分位数 1-5（越高分越高）                | min-max 归一化 0-100（正向）                      |
| 0-100 化    | (score-1)/4×100                         | 直接归一化到 0-100                                |
| 综合分      | 等权 (R+F+M)/3                          | 加权 0.2R+0.3F+0.5M，再叠加意向加成              |
| 最终分      | RFM_Score                               | Final_Score = RFM_Score × (1 + I_Score/500)       |
| 分层        | 中位数切 高/低 × 3 = 8 类               | 硬阈值 8 类基础 + 7 类修正 = 精细化分层            |
| 额外指标    | 无                                      | I 意向度 / Friction 摩擦 / L 忠诚 / 收入分层       |

**打分公式（RFM-I）**：

- `I_Score = 0.5×Time_Spent_Norm + 0.5×Pages_Viewed_Norm`（0-100）
- `Friction = Pages_Viewed / (Purchase_Frequency + 1)`（仅用于洞察与修正分层）
- `L_Score`：订阅且 7 天内登录=3；仅 7 天内登录=2；其余=1
- `RFM_Score = 0.2×R + 0.3×F + 0.5×M`（0-100，M 权重最高）
- `Final_Score = RFM_Score × (1 + I_Score/500)` → 意向加成最高 +20%，上限 120

**分层逻辑**：先按 R/F/M 硬阈值（>60 高 / <40 低）得到 8 类基础人群
（重要价值/重要发展/重要保持/重要挽留/一般发展/一般维持/低价值/一般），
再识别 7 类特殊人群并覆盖：高潜沉睡用户、纠结土豪、犹豫型潜力用户、
隐形活跃者、高潜流失客、核心VIP、羊毛党/低值。

**ROI 测算**（预算 10000 元、券成本 10 元/张，转化率假设可调）：

- 方案A：传统 RFM——RFM_Score 前 20% 用户全体发券
- 方案B：优化 RFI——核心用户（核心VIP + 重要价值）占 80% 预算，
  潜力用户（纠结土豪 / 高潜沉睡 / 犹豫型潜力 / 高潜流失）占 20% 预算
- 边际ROI = (边际收益 − 成本) / 成本 × 100，边际收益 = 人数 × 转化率提升 × 客单价

## 四、本次优化内容（相对原版）

1. **修复 bug**：早期草稿 `rfm_i_models.py` 中 `min_max_normalize` 的
   reverse 方向写反（正向指标被反向归一化）；本工程全局唯一、口径统一。
2. **统一入口**：经典 RFM 与 RFM-I 两条链路整合进同一个 `main.py`，
   可 `--mode` 分别执行；全量结果表同时携带两套分数与分层，便于对比。
3. **配置集中**：权重、阈值、意向加成、预算、转化率、字体全部收敛到
   `Config`（dataclass），改一行即可调参，不再散落在函数里。
4. **健壮性**：经典打分统一走 rank+qcut，避免重复值导致的 qcut 报错；
   输入文件缺失时给出明确提示；路径自动解析，换机器无需改路径。
5. **性能/可读性**：L_Score 由 `apply` 改为向量化 `np.where`；删除了原版
   多余的 `print(df.head())` 调试输出；逐阶段 banner + 计时。
6. **输出规范化**：全部结果统一进 `output/`，Excel 分表（全量明细 /
   分层统计 / 经典RFM / ROI 汇总）+ 5 张图，命名固定、可直接贴进作业。
7. **雷达图改进**：按人数取前 8 类展示（原版按出现顺序截断，且图例带人数）。

## 五、输出清单（output/）

| 文件 | 内容 |
|---|---|
| rfm_analysis_full.xlsx | 全量明细：14 原始列 + 全部衍生列 + 双模型分数 + 两种分层 |
| segment_stats.xlsx | RFM-I 分层统计（人数/占比/平均R/F/M/综合分等） |
| classic_rfm_result.xlsx | 经典 RFM 明细（c_R/c_F/c_M 分数 + 经典分层） |
| roi_summary.xlsx | ROI 方案A/B 对比（人数/成本/收益/边际ROI） |
| eda_distributions.png | EDA 四宫格分布图 |
| correlation_matrix.png | 特征相关性矩阵 |
| segment_radar.png | RFM-I 分层雷达图（前8类） |
| segment_distribution.png | 分层人数分布柱状图 |
| roi_comparison.png | ROI 对比图（边际ROI + 收益构成） |

## 六、数据说明

- 本工程输入 `data/user_personalized_features.xlsx`（1000 行 × 14 列）由
  `extract_raw_data.py` 从既有分析结果表
  `result_r/rfm_analysis_results.xlsx` 中**无损重建**：后者保留了全部原始
  用户字段（User_ID / Age / Gender / Location / Income / Interests /
  Last_Login_Days_Ago / Purchase_Frequency / Average_Order_Value /
  Total_Spending / Product_Category_Preference / Time_Spent_on_Site_Minutes
  / Pages_Viewed / Newsletter_Subscription），仅追加了衍生列。
- 若日后找到原始 `user_personalized_features.xlsx`，直接替换 data 下同名
  文件即可；`main.py` 的校验逻辑会自动识别。
- 数据集为课程演示数据（业务故事设定"1亿+用户"，实际样本 1000 行）。