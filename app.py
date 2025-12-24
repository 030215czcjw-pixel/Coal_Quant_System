import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from filterpy.kalman import KalmanFilter


class TimeSeriesFeatureEngineer:
    def __init__(self, data):
        """
        初始化特征工程处理器
        :param data: pd.DataFrame, 包含原始数据的表格
        """
        self.raw_data = data.copy()

    def _apply_kalman(self, series, Q_val=0.01, R_val=0.1):
        
        """
        内部方法：应用一维卡尔曼滤波进行降噪
        """
        # 确保数据无空值
        vals = series.fillna(method='ffill').fillna(method='bfill').values
        
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.x = np.array([[vals[0]]]) # 初始状态
        kf.F = np.array([[1.]])      # 状态转移矩阵
        kf.H = np.array([[1.]])      # 观测矩阵
        kf.P *= 10.                  # 初始协方差
        kf.R = R_val                 # 测量噪声
        kf.Q = Q_val                 # 过程噪声
        
        filtered_results = []
        for z in vals:
            kf.predict()
            kf.update(z)
            filtered_results.append(kf.x[0, 0])
            
        return filtered_results

    def generate_features(self, n_lag, n_MA, n_D, n_yoy, use_kalman):
        """
        执行特征工程
        :param feature_list: list, 需要生成的特征列表 ["移动平均", "差分", "一阶导数", "二阶导数"]
        :param n_MA: list, 移动平均的窗口列表 [5, 10, 20]
        :param n_D: list, 差分(收益率)的周期列表 [1, 3, 5]
        :param use_kalman: bool, 是否使用卡尔曼滤波预处理数据
        :param kalman_params: dict, 卡尔曼滤波参数
        :param target_col: str, 指定要处理的列名。如果为None，则自动检测第一列数值列。
        :return: pd.DataFrame
        """
        
        # 1. 数据清洗与列选择
        numeric_df = self.raw_data.select_dtypes(include=[np.number])
        if numeric_df.empty:
            # 如果没有识别出数字列，尝试暴力转换
            numeric_df = self.raw_data.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        
        if numeric_df.empty:
            st.error("无法在所选表格中找到数值列，请检查数据格式。")
            return pd.DataFrame()

        target_col = numeric_df.columns[0]
        df = pd.DataFrame(index=self.raw_data.index)
        # 强制转换为 float64，防止 Timestamp 混入
        df['原始数据'] = numeric_df[target_col].astype(float).ffill().bfill()

        # 2. 是否应用卡尔曼滤波
        if use_kalman:
            df['卡尔曼滤波'] = self._apply_kalman(df['原始数据'])
            data_source = df['卡尔曼滤波'] # 后续计算基于滤波后的数据
        else:
            data_source = df['原始数据']

        # 3. 循环生成特征
        if n_lag > 0:
            df[f'滞后{n_lag}'] = data_source.shift(n_lag)
        
        if n_MA > 0:
            df[f'移动平均{n_MA}'] = data_source.rolling(window=n_MA).mean()
                    
        if n_D > 0:
            df[f'差分{n_D}'] = data_source.diff(n_D)
        
        if n_yoy > 0:
            df[f'同比{n_yoy}'] = data_source.pct_change(n_yoy) 
                

        return df

class BayesianStrategyBacktester:
    def __init__(self, stock_data, baseline_data, feature_data, profit_setted, observation_periods, holding_period):
        """
        初始化回测器，执行数据对齐和基础收益率计算。
        """
        self.profit_setted = profit_setted
        self.observation_periods = observation_periods
        self.holding_period = holding_period
        
        # 1. 数据对齐 (Intersection)
        common_dates = stock_data.index.intersection(baseline_data.index).intersection(feature_data.index).sort_values()
        
        # 保存原始数据副本，以便后续使用
        self.feature_data_aligned = feature_data.loc[common_dates].copy()
        
        # 2. 构建基础价格DataFrame
        self.df = pd.DataFrame({
            '股价': stock_data.loc[common_dates, '收盘'],
            '基准': baseline_data.loc[common_dates, 'close'], 
        }, index=common_dates)
        
        # 3. 计算收益率指标 (预处理)
        self.df['股价收益率'] = self.df['股价'].pct_change()
        self.df['基准收益率'] = self.df['基准'].pct_change()
        self.df['超额收益率'] = self.df['股价收益率'] - self.df['基准收益率']
        
        # 计算超额净值曲线
        self.df['超额净值'] = (1 + self.df['超额收益率'].fillna(0)).cumprod()
        
        # 计算未来持有期收益率 (Label)
        # 注意：这里shift是负数，表示读取未来的数据作为当前的标签
        self.df['持有期超额收益率'] = self.df['超额净值'].shift(-holding_period) / self.df['超额净值'] - 1

    def run_strategy(self, feature_cols, strategy_expression):
        """
        执行贝叶斯分析和信号生成
        :param feature_cols: list, 参与计算的特征列名
        :param strategy_expression: str, 策略触发条件的字符串表达式 (例如: "df['RSI'] > 70")
        :return: DataFrame, 包含完整分析结果
        """
        # 使用副本以免污染原始数据
        df = self.df.copy()
        
        # 合并指定的特征列
        for col in feature_cols:
            if col in self.feature_data_aligned.columns:
                df[col] = self.feature_data_aligned[col]
            else:
                print(f"警告: 特征 {col} 不存在于特征数据中")

        # 1. 定义胜率 (Prior Label)
        df['胜率触发'] = (df['持有期超额收益率'] > self.profit_setted).astype(int)
        df['胜率不触发'] = 1 - df['胜率触发']

        # 2. 计算先验概率 P(W) - 使用滚动窗口
        # shift(holding_period) 是为了防止未来函数，确保只用过去的数据计算当前的先验
        df['P(W)'] = df['胜率触发'].rolling(window=self.observation_periods).mean().shift(self.holding_period)
    

        # 3. 执行策略表达式，计算信号 C
        try:
            # 在 eval 的上下文中，df 变量必须可用
            df['信号触发'] = eval(strategy_expression).astype(int)
        except Exception as e:
            print(f"策略表达式错误: {e}") # 替换 st.error 以通用化
            df['信号触发'] = 0

        # 4. 计算条件概率 P(C|W) 和 P(C|not W)
        df['W_and_C'] = ((df['胜率触发'] == 1) & (df['信号触发'] == 1)).astype(int)
        df['notW_and_C'] = ((df['胜率触发'] == 0) & (df['信号触发'] == 1)).astype(int)
        
        # 贝叶斯似然率计算
        rolling_w_c = df['W_and_C'].rolling(self.observation_periods).sum().shift(self.holding_period)
        rolling_w = df['胜率触发'].rolling(self.observation_periods).sum().shift(self.holding_period)
        
        rolling_notw_c = df['notW_and_C'].rolling(self.observation_periods).sum().shift(self.holding_period)
        rolling_notw = df['胜率不触发'].rolling(self.observation_periods).sum().shift(self.holding_period)

        # 避免除以零
        p_c_w = rolling_w_c / rolling_w.replace(0, np.nan)
        p_c_notw = rolling_notw_c / rolling_notw.replace(0, np.nan)
        
        # 5. 计算后验概率 P(W|C)
        # 公式: P(W|C) = P(C|W) * P(W) / [P(C|W)*P(W) + P(C|not W)*P(not W)]
        evidence = p_c_w * df['P(W)'] + p_c_notw * (1 - df['P(W)'])
        df['P(W|C)'] = (p_c_w * df['P(W)']) / evidence.replace(0, np.nan)

        # 6. 生成买入信号
        # 逻辑：后验概率 > 先验概率 且 信号触发 且 (绝对概率>0.5 或 概率动量上升)
        prob_condition = (df['P(W|C)'] > 0.5) | (df['P(W|C)'] > df['P(W|C)'].shift(1) * 0.9)
        improve_condition = df['P(W|C)'] > df['P(W)']
        
        df['买入信号'] = np.where(
            improve_condition & (df['信号触发'] == 1) & prob_condition, 
            1, 0
        )

        # 7. 计算策略净值
        # 仓位逻辑：如果买入，持有 holding_period 天 (这里简化为均摊)
        df['仓位'] = np.where(
            df['买入信号'] == 1, 
            df['信号触发'].shift(1).rolling(self.holding_period).sum() / self.holding_period, 
            0
        )
        
        df['仓位净值'] = (1 + (df['仓位'].shift(1) * df['超额收益率'].fillna(0))).cumprod()
        df['先验仓位净值'] = (1 + (df['P(W)'].shift(1) * df['超额收益率'].fillna(0))).cumprod()

        return df

# ==========================================
# 2. 界面展示逻辑
# ==========================================

class UI:
    def __init__(self):
        pass
    
    def setup_page(self):
        pass
    
st.set_page_config(
            page_title="行业择时回测系统",    # 网页标题 (显示在浏览器标签页)
            page_icon="📈",                # 网页图标 (Favicon，可为 emoji 或图片路径)
            layout="wide",                 # 布局模式 ("centered" 或 "wide")
            initial_sidebar_state="expanded", # 侧边栏初始状态 ("auto", "expanded", "collapsed")
            menu_items={                   # 右上角汉堡菜单的自定义内容
                'Get Help': 'https://github.com/030215czcjw-pixel/Coal_Quant_System',
                'About': "数据可在如下上传和查看\nhttps://docs.google.com/spreadsheets/d/1P3446_9mBi-7qrAMi78F1gHDHGIOCjw-/edit?gid=152940602#gid=152940602\n需要🪜"
            }
        )
st.title("title")
    

# 初始化数据状态
if 'xl_object' not in st.session_state:
    st.session_state['xl_object'] = None
if 'feature_data_after' not in st.session_state:
    st.session_state['feature_data_after'] = None

# --- 侧边栏：数据同步 ---
#st.sidebar.header("数据源同步")
SHEET_ID = "1P3446_9mBi-7qrAMi78F1gHDHGIOCjw-" # 你的谷歌表ID

#@st.cache_resource(ttl=3600)
def fetch_xl_object(sheet_id):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    return pd.ExcelFile(url)

if st.sidebar.button("同步云端表", use_container_width=True):
    with st.spinner("正在扫描云端所有工作表..."):
        st.session_state['xl_object'] = fetch_xl_object(SHEET_ID)
        st.sidebar.success("同步成功！")

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

    #if st.button("加载选定表数据", use_container_width=True):
    df_raw = load_and_clean_feature(xl, feature_selected)
    st.session_state['raw_feature_df'] = df_raw
    st.write(f"{feature_selected} 数据预览：")
    st.dataframe(df_raw)

# --- 侧边栏：参数配置 ---
stock_selected = st.sidebar.selectbox("选择标的", ["中国神华", "综合交易价_CCTD秦皇岛动力煤(Q5500)"])
baseline_selected = st.sidebar.selectbox("选择基准", ["沪深300"])
use_kalman = st.sidebar.checkbox("启用卡尔曼滤波", value=True)

n_lag = st.sidebar.slider("滞后期数", 0, 60, 1)
n_MA = st.sidebar.slider("移动平均窗口", 0, 60, 5)
n_D = st.sidebar.slider("差分期数", 0, 365, 1)
n_yoy = st.sidebar.selectbox("同比期数(1即为环比)", [0, 1, 12, 52, 252])

hp = st.sidebar.slider("持有期（以数据频率为单位）", 1, 365, 5)
op = st.sidebar.slider("观察期（以数据频率为单位）", 1, 365, 60)
profit_target = st.sidebar.number_input("目标超额收益", value=0.0, step=0.01)

s_input = st.sidebar.text_area("策略逻辑 (Python格式)", value="df[''] < 0")

# --- 主界面按钮 ---

if st.button("执行特征工程", use_container_width=True):
    if 'raw_feature_df' not in st.session_state:
        st.error("请先在左侧加载数据！")
    else:
        with st.spinner('特征处理中...'):
            raw_f = st.session_state['raw_feature_df']
            fe_engine = TimeSeriesFeatureEngineer(raw_f )
            processed_fe = fe_engine.generate_features(n_lag, n_MA, n_D, n_yoy, use_kalman) # 执行特征工程
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

            feature_df = st.session_state['feature_data_after']
            
            tester = BayesianStrategyBacktester(
                stock_data=stock_raw,
                baseline_data=baseline_raw,
                feature_data=feature_df,
                profit_setted=profit_target,    # 设定超额收益门槛 2%
                observation_periods=op,# 观察期 60天
                holding_period=hp       # 持有期 5天
            )
            
            df_res = tester.run_strategy(
                feature_cols=feature_df.columns.tolist(),
                strategy_expression=s_input
            )

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
