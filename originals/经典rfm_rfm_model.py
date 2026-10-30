
import pandas as pd
import numpy as np
def load_data(path):
    """加载数据"""
    return pd.read_excel(path)

def calc_rfm_scores(df):
    """
    计算 R/F/M 分数
    R: Last_Login_Days_Ago (越小越好)
    F: Purchase_Frequency (越大越好)
    M: Total_Spending (越大越好)
    """
    rfm = df[['User_ID', 'Last_Login_Days_Ago', 'Purchase_Frequency', 'Total_Spending']].copy()
    rfm.columns = ['User_ID', 'R_raw', 'F_raw', 'M_raw']
    print(rfm.head(10))
    # 分位数打分 1-5
    rfm['R_score'] = pd.qcut(rfm['R_raw'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(float)
    print(rfm.head(10))
    rfm['F_score'] = pd.qcut(rfm['F_raw'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(float)
    print(rfm.head(10))
    rfm['M_score'] = pd.qcut(rfm['M_raw'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(float)
    print(rfm.head(10))
    
    # 归一化到 0-100
    rfm['R_score'] = (rfm['R_score'] - 1) / 4 * 100
    rfm['F_score'] = (rfm['F_score'] - 1) / 4 * 100
    rfm['M_score'] = (rfm['M_score'] - 1) / 4 * 100
    
    # RFM 综合分 (等权)
    rfm['RFM_Score'] = (rfm['R_score'] + rfm['F_score'] + rfm['M_score']) / 3
    
    return rfm


def label_segment(rfm):
    """根据 R/F/M 中位数划分 8 类客户"""
    r_med = rfm['R_score'].median()
    f_med = rfm['F_score'].median()
    m_med = rfm['M_score'].median()
    
    rfm['R_level'] = np.where(rfm['R_score'] >= r_med, '高', '低')
    rfm['F_level'] = np.where(rfm['F_score'] >= f_med, '高', '低')
    rfm['M_level'] = np.where(rfm['M_score'] >= m_med, '高', '低')
    
    segment_map = {
        ('高', '高', '高'): '重要价值客户',
        ('高', '高', '低'): '重要发展客户',
        ('高', '低', '高'): '重要保持客户',
        ('高', '低', '低'): '重要挽留客户',
        ('低', '高', '高'): '一般价值客户',
        ('低', '高', '低'): '一般发展客户',
        ('低', '低', '高'): '一般保持客户',
        ('低', '低', '低'): '一般挽留客户',
    }
    
    rfm['User_Segment'] = rfm.apply(
        lambda r: segment_map[(r['R_level'], r['F_level'], r['M_level'])], axis=1
    )
    return rfm

def summarize(df, rfm):
    # 去掉 Excel 自带的旧打分列（R_Score/F_Score/M_Score/RFM_Score 等），
    # 只保留 User_ID，避免 merge 时与 rfm 新算的同名列冲突变成 RFM_Score_x/_y
    drop_cols = ['User_Segment', 'R_Score', 'F_Score', 'M_Score', 'RFM_Score',
                 'I_Score', 'L_Score', 'Final_Score', 'I_Weight']
    base_cols = [c for c in df.columns if c not in drop_cols]
    df = df[base_cols]
    result = df.merge(rfm[['User_ID', 'R_score', 'F_score', 'M_score', 'RFM_Score', 'User_Segment']], on='User_ID')
    
    summary = (
        result.groupby('User_Segment')
        .agg(
            客户数=('User_ID', 'count'),
            平均R=('R_score', 'mean'),
            平均F=('F_score', 'mean'),
            平均M=('M_score', 'mean'),
            平均RFM=('RFM_Score', 'mean'),
        )
        .reset_index()
    )
    summary['占比'] = summary['客户数'] / summary['客户数'].sum()
    summary = summary.sort_values('客户数', ascending=False)
    
    return result, summary

def main():
    # 加载数据
    df = load_data(r'D:\新建文件夹1\量化学习\数分项目\result_r\rfm_analysis_results.xlsx')
    # 计算 RFM
    rfm = calc_rfm_scores(df)
    # 分层
    rfm = label_segment(rfm)
    # 汇总
    result, summary = summarize(df, rfm)
    # 输出
    print('========== 客户分层汇总 ==========')
    print(summary)
    # 保存结果
    result.to_excel(r'C:\Users\yu354\PyCharmMiscProject\量化学习\数分项目\rfm_result.xlsx', index=False)
    summary.to_excel(r'C:\Users\yu354\PyCharmMiscProject\量化学习\数分项目\rfm_summary.xlsx', index=False)
if __name__ == '__main__':
    main()
