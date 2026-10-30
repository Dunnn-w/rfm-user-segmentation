# -*- coding: utf-8 -*-
"""校验：优化版重算结果 vs 原始分析结果表（result_r/rfm_analysis_results.xlsx）对比。"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MINE = os.path.join(HERE, "output", "rfm_analysis_full.xlsx")
ORIG = r"D:\新建文件夹1\量化学习\数分项目\result_r\rfm_analysis_results.xlsx"

cols = ["I_Score", "Friction", "L_Score", "Income_Level", "Interest_Match",
        "R_Score", "F_Score", "M_Score", "RFM_Score", "I_Weight", "Final_Score",
        "User_Segment"]

a = pd.read_excel(MINE)
b = pd.read_excel(ORIG)
m = a.merge(b[["User_ID"] + cols], on="User_ID", suffixes=("_new", "_old"))
print(f"样本数: {len(m)}")

for c in cols:
    if c in ("Income_Level", "User_Segment"):
        same = (m[f"{c}_new"] == m[f"{c}_old"]).mean()
        print(f"{c:16s} 一致率: {same * 100:.2f}%")
    else:
        diff = (m[f"{c}_new"] - m[f"{c}_old"]).abs().max()
        print(f"{c:16s} 最大绝对差: {diff:.10f}")