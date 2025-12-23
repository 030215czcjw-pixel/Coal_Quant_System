import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ==========================================
# 1. 核心算法逻辑 (增强了鲁棒性)
# ==========================================

def apply_filterpy_kalman(series, Q_val=0.01, R_val=0.1):
    from filterpy.kalman import KalmanFilter
    # 确保传入的是 numpy 数组且无空值
    vals = series.fillna(method='ffill').fillna(method='bfill').values
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[vals[0]]])
    kf.F = np.array([[1.]])
    kf.H = np.array([[1.]])
    kf.P *= 10.
    kf.R = R_val
    kf.Q = Q_val
    
    filtered_results = []
    for z in vals:
        kf.predict()
        kf.update(z)
        filtered_results.append(kf.x[0, 0])
    return filtered_results

def FE(original_feature, n_MA, n_D, Y_window, Q_window, feature_name, use_kalman):
    """
    特征工程：智能识别数值列，避开日期列导致的编码错误
    """
    # 1. 自动筛选数值列 (避开日期类型)
    numeric_df = original_feature.select_dtypes(include=[np.number])
    if numeric_df.empty:
        # 如果没有识别出数字列，尝试暴力转换
        numeric_df = original_feature.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    
    if numeric_df.empty:
        st.error("无法在所选表格中找到数值列，请检查数据格式。")
        return pd.DataFrame()

    target_col = numeric_df.columns[0]
    df = pd.DataFrame(index=original_feature.index)
    # 强制转换为 float64，防止 Timestamp 混入
    df['原始数据'] = numeric_df[target_col].astype(float).ffill().bfill()

    if use_kalman:
        df['卡尔曼滤波'] = apply_filterpy_kalman(df['原始数据'])
        data = df['卡尔曼滤波']
    else:
        data = df['原始数据']
        
    for op in feature_name:
        if op == "移动平均":
            for ma in n_MA:
                df[f'移动平均{ma}'] = data.rolling(window=ma).mean()
        if op == "差分":
            for d in n_D:
                df[f'差分{d}'] = data.pct_change(periods=d)
        if op == "一阶导数":
            df['一阶导数'] = data.diff(1)
        if op == "二阶导数":
            df['二阶导数'] = data.diff(1).diff(1)
    
    return df

def set_price_data(stock_data, baselinedata, feature_data, holding_period):
    # 确保索引对齐
    common_dates = stock_data.index.intersection(baselinedata.index).intersection(feature_data.index).sort_values()
    
    price_data = pd.DataFrame({
        '股价': stock_data.loc[common_dates, '收盘'],
        '基准': baselinedata.loc[common_dates, 'close'],
    }, index=common_dates)
    
    price_data['股价收益率'] = price_data['股价'].pct_change()
    price_data['基准收益率'] = price_data['基准'].pct_change()
    price_data['超额收益率'] = price_data['股价收益率'] - price_data['基准收益率']
    
    # 计算净值
    price_data['超额净值'] = (1 + price_data['超额收益率'].fillna(0)).cumprod()
    price_data['持有期超额收益率'] = price_data['超额净值'].shift(-holding_period) / price_data['超额净值'] - 1
    
    return price_data

def bayesian_analysis(price_data, feature_data, profit_setted, observation_periods, holding_period, f, s):
    common_dates = price_data.index.intersection(feature_data.index).sort_values()
    df = price_data.loc[common_dates].copy()
    
    for col in f:
        df[col] = feature_data.loc[common_dates, col]
    
    df['胜率触发'] = (df['持有期超额收益率'] > profit_setted).astype(int)
    df['胜率不触发'] = 1 - df['胜率触发']
    
    # 贝叶斯核心计算
    pw_early = df['胜率触发'].rolling(window=observation_periods).mean().shift(holding_period)
    pw_late = df['胜率触发'].rolling(window=observation_periods).mean().shift(holding_period + 1)
    cutoff = observation_periods + holding_period
    df['P(W)'] = pw_early
    if len(df) > cutoff:
        df.iloc[cutoff:, df.columns.get_loc('P(W)')] = pw_late.iloc[cutoff:]
    
    # 安全执行策略逻辑
    try:
        df['信号触发'] = eval(s).astype(int)
    except Exception as e:
        st.error(f"策略表达式错误: {e}")
        df['信号触发'] = 0

    # 条件概率 P(C|W) 和 P(C|not W)
    shift_n = holding_period + 1
    df['W_and_C'] = ((df['胜率触发'] == 1) & (df['信号触发'] == 1)).astype(int)
    df['notW_and_C'] = ((df['胜率触发'] == 0) & (df['信号触发'] == 1)).astype(int)
    
    p_c_w = (df['W_and_C'].rolling(observation_periods).sum().shift(shift_n) / 
             df['胜率触发'].rolling(observation_periods).sum().shift(shift_n))
    p_c_notw = (df['notW_and_C'].rolling(observation_periods).sum().shift(shift_n) / 
                df['胜率不触发'].rolling(observation_periods).sum().shift(shift_n))
    
    df['P(W|C)'] = (p_c_w * df['P(W)']) / (p_c_w * df['P(W)'] + p_c_notw * (1 - df['P(W)']))
    
    # 信号生成与仓位
    df['买入信号'] = np.where(
        (df['P(W|C)'] > df['P(W)']) & (df['信号触发'] == 1) & 
        ((df['P(W|C)'] > 0.5) | (df['P(W|C)'] > df['P(W|C)'].shift(1)*0.9)), 1, 0
    )
    df['仓位'] = np.where(df['买入信号'] == 1, 
                        df['信号触发'].shift(1).rolling(holding_period).sum() / holding_period, 0)
    
    df['仓位净值'] = (1 + (df['仓位'].shift(1) * df['超额收益率'].fillna(0))).cumprod()
    df['先验仓位净值'] = (1 + (df['P(W)'].shift(1) * df['超额收益率'].fillna(0))).cumprod()
    
    return df

# ==========================================
# 2. 界面展示逻辑
# ==========================================

st.set_page_config(page_title="煤炭择时回测系统", layout="wide")
st.title("煤炭行业贝叶斯择时回测平台")

# 初始化数据状态
if 'xl_object' not in st.session_state:
    st.session_state['xl_object'] = None
if 'feature_data_after' not in st.session_state:
    st.session_state['feature_data_after'] = None

# --- 侧边栏：数据同步 ---
st.sidebar.header("📁 数据源同步")
SHEET_ID = "1P3446_9mBi-7qrAMi78F1gHDHGIOCjw-" # 你的谷歌表ID

#@st.cache_resource(ttl=3600)
def fetch_xl_object(sheet_id):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    return pd.ExcelFile(url)

if st.sidebar.button("🔄 同步云端表结构"):
    with st.spinner("正在扫描云端所有工作表..."):
        st.session_state['xl_object'] = fetch_xl_object(SHEET_ID)
        st.success("同步成功！")

# 只有同步后才显示下拉菜单
if st.session_state['xl_object'] is not None:
    xl = st.session_state['xl_object']
    feature_selected = st.sidebar.selectbox("选择特征维度", xl.sheet_names)
    
    # 核心数据加载函数：带日期自动识别
    def load_and_clean_feature(xl_obj, sheet_name):
        df = xl_obj.parse(sheet_name)
        # 自动寻找日期列并设为索引
        for col in df.columns:
            if '日期' in str(col) or 'Date' in str(col) or 'time' in str(col).lower():
                df[col] = pd.to_datetime(df[col])
                df.set_index(col, inplace=True)
                break
        return df

    if st.sidebar.button("加载选定表数据"):
        df_raw = load_and_clean_feature(xl, feature_selected)
        st.session_state['raw_feature_df'] = df_raw
        st.write(f"✅ {feature_selected} 数据预览：")
        st.dataframe(df_raw.head())

# --- 侧边栏：参数配置 ---
st.sidebar.divider()
stock_selected = st.sidebar.selectbox("选择标的", ["中国神华"])
baseline_selected = st.sidebar.selectbox("选择基准", ["沪深300"])
use_kalman = st.sidebar.checkbox("启用卡尔曼滤波", value=True)
features_op = st.sidebar.multiselect("操作算子", ["移动平均", "差分", "一阶导数", "二阶导数"], default=["一阶导数"])

n_MA = st.sidebar.slider("移动平均窗口", 1, 60, 5)
n_D = st.sidebar.slider("差分期数", 1, 10, 1)
hp = st.sidebar.slider("持有期（以数据频率为单位）", 1, 365, 5)
op = st.sidebar.slider("观察期（以数据频率为单位）", 1, 365, 60)
profit_target = st.sidebar.number_input("目标超额收益", value=0.0, step=0.01)

s_input = st.sidebar.text_area("策略逻辑 (Python格式)", value="df['一阶导数'] < 0")

# --- 主界面按钮 ---

if st.button("执行特征工程", use_container_width=True):
    if 'raw_feature_df' not in st.session_state:
        st.error("请先在左侧加载数据！")
    else:
        with st.spinner('特征处理中...'):
            raw_f = st.session_state['raw_feature_df']
            processed_fe = FE(raw_f, [n_MA], [n_D], 12, 12, features_op, use_kalman)
            st.session_state['feature_data_after'] = processed_fe
            st.success("特征工程完成！")
            st.dataframe(processed_fe)


if st.button("执行回测分析", use_container_width=True):
    if st.session_state['feature_data_after'] is None:
        st.error("请先执行特征工程！")
    else:
        with st.spinner('贝叶斯回测中...'):
            # 读取本地股票数据 (需确保文件在同目录下)
            try:
                stock_raw = pd.read_excel('stock_data.xlsx', sheet_name=stock_selected, index_col='日期', parse_dates=True)
                baseline_raw = pd.read_excel('stock_data.xlsx', sheet_name=baseline_selected, index_col='date', parse_dates=True)
            except:
                st.error("本地 stock_data.xlsx 读取失败，请检查文件。")
                st.stop()

            fe_data = st.session_state['feature_data_after']
            p_data = set_price_data(stock_raw, baseline_raw, fe_data, hp)
            df_res = bayesian_analysis(p_data, fe_data, profit_target, op, hp, fe_data.columns.tolist(), s_input)

            # --- 结果展示 ---
            final_nav = df_res['仓位净值'].iloc[-1]
            prior_nav = df_res['先验仓位净值'].iloc[-1]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("策略净值", f"{final_nav:.3f}", f"{(final_nav-1):.2%}")
            c2.metric("先验净值", f"{prior_nav:.3f}", f"{(prior_nav-1):.2%}", delta_color="off")
            c3.metric("超额增益", f"{(final_nav-prior_nav):.2%}")

            # Plotly 图表
            fig = make_subplots(rows=2, cols=2, subplot_titles=("胜率修正", "净值表现", "信号触发", "实时仓位"),
                               specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                    [{"secondary_y": False}, {"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['P(W)'], name='先验', line=dict(color='orange')), 1, 1)
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['P(W|C)'], name='后验', line=dict(color='grey', dash='dot')), 1, 1)
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['仓位净值'], name='策略仓位净值', line=dict(color='red')), 1, 2)
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['先验仓位净值'], name='先验仓位净值', line=dict(color='grey')), 1, 2)

            fig.add_trace(go.Scatter(
                x=df_res.index, 
                y=df_res['超额净值'], 
                name='超额净值', 
                line=dict(color='blue', width=1.5)
            ), 2, 1)
            
            # 再画信号背景
            # 技巧：把信号 y 轴放大到超额净值的范围，或者直接用 yaxis2
            fig.add_trace(go.Scatter(
                x=df_res.index, 
                y=df_res['信号触发'], 
                name='触发脉冲', 
                fill='tozeroy', 
                line=dict(width=0),
                fillcolor='rgba(255, 165, 0, 0.2)', # 浅橙色背景
            ), 2, 1)
            
            fig.add_trace(
                go.Scatter(
                    x=df_res.index, 
                    y=df_res['超额净值'], 
                    name='超额净值', 
                    line=dict(color='blue', width=2),
                    hovertemplate='日期: %{x}<br>超额净值: %{y:.4f}<extra></extra>'
                ), 
                row=2, col=2, secondary_y=False
            )
            
            # 2. 绘制仓位（作为次 Y 轴阴影，使用阶梯线）
            fig.add_trace(
                go.Scatter(
                    x=df_res.index, 
                    y=df_res['仓位'], 
                    name='策略仓位', 
                    fill='tozeroy', 
                    # 核心优化：使用阶梯线（hv），真实还原调仓的离散跳变
                    line_shape='hv', 
                    line=dict(color='rgba(255, 165, 0, 0.8)', width=1), 
                    # 浅橙色填充，不遮挡背景净值线
                    fillcolor='rgba(255, 165, 0, 0.2)', 
                    hovertemplate='日期: %{x}<br>当前仓位: %{y:.2f}<extra></extra>'
                ), 
                row=2, col=2, secondary_y=True
            )
            
            # 3. 更新 Y 轴设置，确保尺度专业
            fig.update_yaxes(title_text="净值水平", secondary_y=False, row=2, col=2)
            fig.update_yaxes(title_text="仓位权重", range=[0, 1.1], secondary_y=True, row=2, col=2)
            
            fig.update_layout(height=700, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
