import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def apply_filterpy_kalman(series, Q_val=0.01, R_val=0.1):
    from filterpy.kalman import KalmanFilter
    
    # 1. 初始化滤波器
    # dim_x=1: 状态变量为1维（位置）
    # dim_z=1: 观测变量为1维（测量值）
    kf = KalmanFilter(dim_x=1, dim_z=1)
    
    # 2. 配置参数
    kf.x = np.array([[series.iloc[0]]])  # 初始状态：设置为第一个观测值
    kf.F = np.array([[1.]])         # 状态转移矩阵
    kf.H = np.array([[1.]])         # 观测矩阵
    kf.P *= 10.                     # 初始协方差，表示对初始值的不确定性
    kf.R = R_val                    # 测量噪声方差
    kf.Q = Q_val                    # 过程噪声方差
    
    filtered_results = []
    
    # 3. 遍历数据并更新
    for z in series:
        kf.predict()         # 预测下一时刻状态
        kf.update(z)         # 根据观测值更新估计
        filtered_results.append(kf.x[0, 0])
        
    return filtered_results

def calculate_seasonal_zscore_walk_forward(df_input, value_col='原始数据'):
    """
    计算滚动季节性 Z-Score (无未来函数版)
    逻辑：当前周的均值和标准差仅由历史同周数据决定
    """
    df = df_input.copy()
    df['week'] = df.index.isocalendar().week
    df['year'] = df.index.year
    
    # 初始化结果列
    df['seasonal_z'] = np.nan
    
    # 按照周进行分组处理
    for week_num, group in df.groupby('week'):
        # 核心：计算该周在历史上的滚动均值和标准差 (expanding)
        # shift(1) 是关键：确保今天计算 Z-Score 时，用的是去年及以前的统计量
        rolling_mean = group[value_col].expanding().mean().shift(1)
        rolling_std = group[value_col].expanding().std().shift(1)
        
        # 计算 Z-Score
        z_scores = (group[value_col] - rolling_mean) / rolling_std
        df.loc[group.index, 'seasonal_z'] = z_scores
        
    return df['seasonal_z']

def FE(original_feature, n_MA, n_D, Y_window, Q_window, feature_name, use_kalman):
    # 1. 准备基础 DataFrame 并保留原始索引
    df = pd.DataFrame(index=original_feature.index)
    df['原始数据'] = original_feature.iloc[:, 0]
    if use_kalman:
        df['卡尔曼滤波'] = apply_filterpy_kalman(df['原始数据'], Q_val=0.01, R_val=0.1)
        data = df['卡尔曼滤波']
    else:
        data = df['原始数据']
    for _ in feature_name:
        if _ == "移动平均":
            for ma in n_MA:
                df[f'移动平均{ma}'] = data.rolling(window=ma).mean()
        if _ == "差分":
            for d in n_D:
                df[f'差分{d}'] = data.pct_change(periods=d)
        if _ == "一阶导数":
            df['一阶导数'] = data.diff(1)
        if _ == "二阶导数":
            df['二阶导数'] = data.diff(1).diff(1)
    
        #滚动年度累计
        #df['滚动年度累计'] = original_feature.iloc[:, 0].rolling(window=Y_window, min_periods=Y_window).sum()

        #滚动年度环比
        #df['滚动年度环比'] = original_feature.iloc[:, 0]/ original_feature.iloc[:, 0].shift(Q_window) - 1
        
        #滚动年度同比
        #df['滚动年度同比'] = original_feature.iloc[:, 1].pct_change(periods=Y_window)
    
    
    return df

def visualize(df, s, stock_name, feature_sheet_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.dates as mdates

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False


    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title('先验胜率')
    plt.plot(df.index, df['P(W)'], label='先验胜率', color='orange')
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=12*30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    plt.subplot(2, 2, 2)
    plt.title('后验胜率对先验胜率的修正')
    #plt.plot(df.index, df['超额净值'], label='超额净值', color='blue')
    plt.plot(df.index, df['P(W)'], label='先验胜率', color='orange')
    plt.plot(df.index, df['P(W|C)'], label='后验胜率', color='grey')
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=12*30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    plt.subplot(2, 2, 3)
    plt.title('历史条件触发情况')
    plt.plot(df.index, df['超额净值'], label='超额净值', color='blue')
    plt.plot(df.index, df['信号触发'], label='信号触发', color='orange')
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=12*30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    plt.subplot(2, 2, 4)
    plt.title('观测条件增益情况')
    #定义归一化函数
    def min_max_scale(series):
        return (series - series.min()) / (series.max() - series.min())
    
    plt.plot(df.index, min_max_scale(df['仓位净值']), label='信号策略净值', color='orange')
    plt.plot(df.index, min_max_scale(df['先验仓位净值']), label='先验策略净值', color='grey')
    plt.plot(df.index, min_max_scale(df['仓位']), label='信号策略仓位', color='blue')
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.3)
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=12*30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "output_pics")
    
    # 确保文件夹存在，如果没有就创建一个
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- 关键修复：清洗文件名 ---
    # 替换掉 / (路径符), : (保留符), 以及可能引起问题的引号和括号
    clean_s = s.replace('/', '_div_').replace(':', '_').replace(' ', '').replace("'", "").replace("[", "").replace("]", "")
    clean_stock = stock_name.replace(':', '_')
    
    filename = f"{clean_stock}_{feature_sheet_name}_{clean_s}.png"
    save_path = os.path.join(save_dir, filename)
    # --------------------------

    plt.savefig(save_path)
    plt.close()
    print(f'图像保存成功: {filename}')
    
def set_price_data(stock_data, baselinedata, feature_data, holding_period): #构建价格数据
    
    #处理时间差异，有些日期可能缺失，取交集
    common_dates = stock_data.index.intersection(baselinedata.index).sort_values()
    stock_filtered = stock_data.loc[common_dates]
    baseline_filtered = baselinedata.loc[common_dates]
    
    price_data = pd.DataFrame({
        '日期': common_dates,
        '股价': stock_filtered['收盘'],
        '基准': baseline_filtered['close'],
    }, index=common_dates)
    
    price_data = price_data[~price_data.index.duplicated(keep='first')]
    feature_data = feature_data[~feature_data.index.duplicated(keep='first')]
    
    common_dates2 = price_data.index.intersection(feature_data.index).sort_values()
    price_data = price_data.loc[common_dates2]
    
    price_data['股价收益率'] = price_data['股价'].pct_change()
    price_data['基准收益率'] = price_data['基准'].pct_change()
    price_data['超额收益率'] = price_data['股价收益率'] - price_data['基准收益率']
    
    price_data['股价净值'] = (1 + price_data['股价收益率']).cumprod()
    price_data.iloc[0, price_data.columns == '股价净值'] = 1
    
    price_data['基准净值'] = (1 + price_data['基准收益率']).cumprod()
    price_data.iloc[0, price_data.columns == '基准净值'] = 1
    
    price_data['超额净值'] = (1 + price_data['超额收益率']).cumprod()
    price_data.iloc[0, price_data.columns == '超额净值'] = 1
    
    price_data['持有期绝对收益'] = price_data['股价净值'].shift(-holding_period) - price_data['股价净值']
    price_data['持有期超额收益率'] = price_data['超额净值'].shift(-holding_period) / price_data['超额净值'] - 1
    
    price_data.to_excel('prcie_data.xlsx', index=False)
    return price_data

def bayesian_analysis(price_data, feature_data, profit_setted, observation_periods, holding_period, f, s): #进行贝叶斯测算
    
    price_data = price_data[~price_data.index.duplicated(keep='first')]
    feature_data = feature_data[~feature_data.index.duplicated(keep='first')]
    
    common_dates = price_data.index.intersection(feature_data.index).sort_values()
    price_filtered = price_data.loc[common_dates]
    feature_filtered = feature_data.loc[common_dates]
    
    df=pd.DataFrame({
        '日期': common_dates,
        '股价': price_filtered['股价'],
        '基准': price_filtered['基准'],
        '超额净值': price_filtered['超额净值'],
        '超额收益率': price_filtered['超额收益率'],
        '持有期超额收益率': price_filtered['持有期超额收益率']
    }, index=common_dates)
    
    #读入特征
    for _ in f:
        df[f'{_}'] = feature_filtered[_] 
    
    df['胜率触发'] = df['持有期超额收益率'].apply(lambda x: 1 if x > profit_setted else 0)
    df['胜率不触发'] = (df['胜率触发'] == 0).astype(int)
    
    #excel中方式有区别，个人认为还是如下的代码正确
    #df['P(W)'] = df['胜率触发'].rolling(window=observation_periods, min_periods=1).mean().shift(holding_period)
    # 但为了复现excel结果，采用如下方式：
    # 1. 计算两个版本的位移
    # 版本 A: 适用于早期的 Shift 12
    pw_early = df['胜率触发'].rolling(window=observation_periods, min_periods=1).mean().shift(holding_period)
    # 版本 B: 适用于稳定期的 Shift 13 (即 holding_period + 1)
    pw_late = df['胜率触发'].rolling(window=observation_periods, min_periods=1).mean().shift(holding_period + 1)
    # 2. 定义切换点
    # 切换点 = 统计期 (100) + 持有期 (12) = 112
    cutoff_index = observation_periods + holding_period
    # 3. 混合拼接
    # 如果 DataFrame 索引是默认的数字索引 (0, 1, 2...)
    df['P(W)'] = pw_early
    df.iloc[cutoff_index:, df.columns.get_loc('P(W)')] = pw_late.iloc[cutoff_index:]
    
    df['信号触发'] = (eval(s)).astype(int) #这里写触发条件
    
    df['W and C'] = ((df['胜率触发'] == 1) & (df['信号触发'] == 1)).astype(int)
    df['notW and C'] = ((df['胜率触发'] == 0) & (df['信号触发'] == 1)).astype(int)
    df['P(C|W)'] = (df['W and C'].rolling(window=observation_periods, min_periods=1).sum().shift(holding_period+1) / df['胜率触发'].rolling(window=observation_periods, min_periods=1).sum().shift(holding_period+1))
    df['P(C|notW)'] = df['notW and C'].rolling(window=observation_periods, min_periods=1).sum().shift(holding_period+1) / df['胜率不触发'].rolling(window=observation_periods, min_periods=1).sum().shift(holding_period+1)
    df['P(W|C)'] = (df['P(C|W)'] * df['P(W)']) / (df['P(C|W)'] * df['P(W)'] + df['P(C|notW)'] * (1 - df['P(W)']))
    
    df['买入信号'] = np.where(
        (df['P(W|C)'] > df['P(W)']) & (df['信号触发'] == 1) & ((df['P(W|C)'] > 0.5) | (df['P(W|C)'] > df['P(W|C)'].shift(1)*0.9)),
        1,
        0
    )
    
    #仓位由过去持有期内信号触发次数决定
    df['仓位'] = np.where(
        (df['买入信号'] == 1),
        df['信号触发'].shift(1).rolling(window=holding_period).sum() / holding_period,
        0
    )
    
    strategy_returns = df['仓位'].shift(1) * df['超额收益率']
    df['仓位净值'] = (1 + strategy_returns).cumprod()
    df['仓位净值'] = df['仓位净值'].fillna(1)
    
    strategy_returns2 = df['P(W)'].shift(1) * df['超额收益率']
    df['先验仓位净值'] = (1 + strategy_returns2).cumprod()
    df['先验仓位净值'] = df['先验仓位净值'].fillna(1)
    
    df.to_excel('bayes.xlsx', index=False)
    return df


st.set_page_config(page_title="煤炭择时因子回测系统", layout="wide")
st.title("煤炭行业贝叶斯择时回测平台")

# 初始化 Session State 用于跨按钮保存特征数据
if 'feature_data_after' not in st.session_state:
    st.session_state['feature_data_after'] = None

st.sidebar.header("策略参数配置")
stock_selected = st.sidebar.selectbox("选择标的", ["中国神华"])
baseline_selected = st.sidebar.selectbox("选择基准", ["沪深300"])
feature_selected = st.sidebar.selectbox("特征维度", ["可用天数", "沿海煤炭运价指数", "北方港合计库存量"])
feature_frequence = st.sidebar.selectbox("特征频率", ["日", "周", "月"])
use_kalman = st.sidebar.checkbox("启用卡尔曼滤波", value=True)
features_op = st.sidebar.multiselect("对所选特征进行的操作", ["移动平均", "差分", "一阶导数", "二阶导数"], default=["一阶导数", "二阶导数"])

n_MA = st.sidebar.slider("移动平均数", 1, 365, 5)
n_D = st.sidebar.slider("差分数", 1, 10, 1)
hp = st.sidebar.slider("持有期 (holding_period)", 1, 60, 2)
op = st.sidebar.slider("观察期 (observation_periods)", 30, 250, 30)
profit_target = st.sidebar.number_input("目标超额收益率 (profit_setted)", value=0.0, step=0.01)

s_input = st.sidebar.text_area("策略逻辑 (Python 表达式)", 
                              value="(df['卡尔曼滤波'].diff(1) < 0) & (df['卡尔曼滤波'].diff(1).diff(1) < 0)")

@st.cache_data
def load_data(stock, baseline, feature):
    stock_df = pd.read_excel('stock_data.xlsx', sheet_name=stock, index_col='日期', parse_dates=True)
    baseline_df = pd.read_excel('stock_data.xlsx', sheet_name=baseline, index_col='date', parse_dates=True)
    feature_df = pd.read_excel('动力煤特征.xlsx', sheet_name=feature, index_col='日期', parse_dates=True)
    return stock_df, baseline_df, feature_df

# --- 按钮 1：执行特征工程 ---
if st.button("🛠 执行特征工程"):
    with st.spinner('特征处理中...'):
        stock_raw, baseline_raw, feature_raw = load_data(stock_selected, baseline_selected, feature_selected)
        
        # 运行 FE 函数并存入 session_state
        processed_fe = FE(feature_raw, 
                          n_MA=[n_MA], 
                          n_D=[n_D], 
                          Y_window=12, 
                          Q_window=12, 
                          feature_name=features_op,
                          use_kalman=use_kalman)
        
        st.session_state['feature_data_after'] = processed_fe
        
        st.success(f"特征工程完成！生成列：{processed_fe.columns.tolist()}")
        st.subheader("特征工程结果预览")
        st.dataframe(processed_fe)

# --- 按钮 2：执行回测分析 ---
if st.button("🚀 执行回测分析"):
    # 如果用户没点第一个按钮，自动运行一次 FE
    if st.session_state['feature_data_after'] is None:
        stock_raw, baseline_raw, feature_raw = load_data(stock_selected, baseline_selected, feature_selected)
        st.session_state['feature_data_after'] = FE(feature_raw, [n_MA], [n_D], 12, 12, features_op, use_kalman)
    
    with st.spinner('贝叶斯回测计算中...'):
        stock_raw, baseline_raw, _ = load_data(stock_selected, baseline_selected, feature_selected)
        fe_data = st.session_state['feature_data_after']
        
        # 1. 构建价格数据
        price_data = set_price_data(stock_raw, baseline_raw, fe_data, holding_period=hp)
        
        # 2. 贝叶斯分析
        df_result = bayesian_analysis(
            price_data, 
            fe_data, 
            profit_setted=profit_target, 
            observation_periods=op, 
            holding_period=hp, 
            f=fe_data.columns.tolist(), 
            s=s_input
        )

        st.success("回测完成！")
        
        # 1. 计算核心指标
        # 最终策略净值
        final_strategy_nav = df_result['仓位净值'].iloc[-1]
        # 最终基准净值
        final_prior_nav = df_result['先验仓位净值'].iloc[-1]

        # 计算收益率
        strategy_return = (final_strategy_nav - 1)
        prior_return = (final_prior_nav - 1)
        excess_return = strategy_return - prior_return # 超额收益

        # 2. 使用列布局并行显示
        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                label="策略最终净值", 
                value=f"{final_strategy_nav:.3f}", 
                delta=f"{strategy_return:.2%}"
            )

        with m2:
            st.metric(
                label="先验基准净值", 
                value=f"{final_prior_nav:.3f}", 
                delta=f"{prior_return:.2%}",
                delta_color="off" # 基准的变化通常设为灰色
            )

        with m3:
            # 超额收益，如果是正的就显示绿色增量
            st.metric(
                label="贝叶斯超额增益", 
                value=f"{excess_return:.2%}", 
                delta=f"{(excess_return):.2%}"
            )
        
        st.divider() # 添加分割线

        # 3. Plotly 交互图表 (修复 alpha 后的版本)
        st.subheader("回测详细数据看板")

        fig = make_subplots(
            rows=2, cols=2, 
            subplot_titles=("胜率修正曲线", "策略净值表现", "信号触发点位", "实时仓位变动")
        )

        # 子图 1
        fig.add_trace(go.Scatter(x=df_result.index, y=df_result['P(W)'], name='先验胜率', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_result.index, y=df_result['P(W|C)'], name='后验胜率', line=dict(color='grey', dash='dot')), row=1, col=1)

        # 子图 2
        fig.add_trace(go.Scatter(x=df_result.index, y=df_result['仓位净值'], name='策略净值', line=dict(color='red')), row=1, col=2)
        fig.add_trace(go.Scatter(x=df_result.index, y=df_result['先验仓位净值'], name='基准净值', line=dict(color='grey')), row=1, col=2)

        # 子图 3
        fig.add_trace(go.Scatter(x=df_result.index, y=df_result['超额净值'], name='超额净值', line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Bar(x=df_result.index, y=df_result['信号触发'], name='信号', marker_color='orange', opacity=0.3), row=2, col=1)

        # 子图 4 (已修复 alpha 错误)
        fig.add_trace(go.Scatter(
            x=df_result.index, 
            y=df_result['仓位'], 
            name='实时仓位', 
            fill='tozeroy', 
            line=dict(color='rgba(0, 0, 255, 0.5)'), # 使用 rgba 替代 alpha
            opacity=0.4
        ), row=2, col=2)

        fig.update_layout(height=700, hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
