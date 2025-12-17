"""
VALORANT Champions 多年度对比数据可视化分析系统
支持2024年 vs 2025年数据对比分析
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import os

# ==================== 页面配置 ====================
st.set_page_config(
    layout="wide",
    page_title="VALORANT Champions 多年度对比分析",
    page_icon="🎮",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS样式 ====================
st.markdown("""
<style>
    :root {
        --valorant-red: #FF4655;
        --valorant-blue: #5865F2;
        --valorant-cyan: #00D9FF;
        --gold: #FFD700;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #FF4655 0%, #00D9FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .year-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0 0.5rem;
    }
    
    .year-2024 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .year-2025 {
        background: linear-gradient(135deg, #FF4655 0%, #FF8C42 100%);
        color: white;
    }
    
    .insight-box {
        background-color: #f0f8ff;
        border-left: 5px solid #FF4655;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .comparison-highlight {
        background: linear-gradient(90deg, rgba(255,70,85,0.1) 0%, rgba(0,217,255,0.1) 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 全局颜色主题配置 ====================
COLOR_2024 = "#667eea"   # 2024年主色（蓝紫）
COLOR_2025 = "#FF4655"   # 2025年主色（红橙）
COLOR_POSITIVE = "#22c55e"  # 正向变化/进步
COLOR_NEGATIVE = "#ef4444"  # 负向变化/退步
COLOR_NEUTRAL = "#6b7280"   # 中性/辅助
SCALE_2024 = "Blues"        # 2024连续色板
SCALE_2025 = "Reds"         # 2025连续色板
DIVERGING_SCALE = "RdBu_r"   # 发散色板（正负变化）

# ==================== 数据加载模块 ====================
# 辅助函数：给图表坐标轴加粗
def bold_axis_labels(fig, xlabel=None, ylabel=None):
    """给Plotly图表的x轴和y轴标签加粗"""
    if xlabel:
        fig.update_xaxes(title_text=f"<b>{xlabel}</b>")
    if ylabel:
        fig.update_yaxes(title_text=f"<b>{ylabel}</b>")
    return fig

@st.cache_data
def load_multi_year_data():
    """加载多年度数据"""
    data_2024 = None
    data_2025 = None
    merged_data = None
    
    # 尝试加载2024年数据
    if os.path.exists("data/2024/processed/2024_all_players.csv"):
        try:
            players_2024 = pd.read_csv("data/2024/processed/2024_all_players.csv")
            data_2024 = preprocess_data(players_2024, '2024')
        except Exception as e:
            st.warning(f"2024年数据加载失败: {str(e)}")
    
    # 尝试加载2025年数据
    if os.path.exists("data/2025/processed/2025_all_players.csv"):
        try:
            players_2025 = pd.read_csv("data/2025/processed/2025_all_players.csv")
            data_2025 = preprocess_data(players_2025, '2025')
        except Exception as e:
            st.warning(f"2025年数据加载失败: {str(e)}")
    
    # 尝试加载旧版数据作为2025年数据
    if data_2025 is None and os.path.exists("data/processed/2025_all_players.csv"):
        try:
            players_2025 = pd.read_csv("data/processed/2025_all_players.csv")
            data_2025 = preprocess_data(players_2025, '2025')
        except Exception as e:
            st.warning(f"默认数据加载失败: {str(e)}")
    
    # 尝试加载合并数据
    if os.path.exists("data/merged/all_players_merged.csv"):
        try:
            merged_data = pd.read_csv("data/merged/all_players_merged.csv")
            merged_data = preprocess_data(merged_data, None)
        except Exception as e:
            st.warning(f"合并数据加载失败: {str(e)}")
    
    return data_2024, data_2025, merged_data

def preprocess_data(df, year):
    """数据预处理"""
    # 确保有year列
    if 'year' not in df.columns and year is not None:
        df['year'] = year
    
    # 数据类型转换
    numeric_columns = [
        'Rating 2.0', 'Average Combat Score', 'Kills', 'Deaths', 'Assists',
        'Kills - Deaths', 'Kill, Assist, Trade, Survive %', 
        'Average Damage per Round', 'Headshot %', 'First Kills', 'First Deaths'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 计算衍生指标
    df['KDA'] = (df['Kills'] + df['Assists']) / df['Deaths'].replace(0, 1)
    df['FK_FD_Diff'] = df['First Kills'] - df['First Deaths']
    
    # 重命名列
    df = df.rename(columns={
        'Rating 2.0': 'Rating',
        'Average Combat Score': 'ACS',
        'Kills - Deaths': 'KD_Diff',
        'Kill, Assist, Trade, Survive %': 'KAST',
        'Average Damage per Round': 'ADR',
        'Headshot %': 'HS_Percent',
        'First Kills': 'FK',
        'First Deaths': 'FD'
    })
    
    return df

data_2024, data_2025, merged_data = load_multi_year_data()

# 检查数据可用性
has_2024 = data_2024 is not None and len(data_2024) > 0
has_2025 = data_2025 is not None and len(data_2025) > 0
has_both = has_2024 and has_2025

if not has_2024 and not has_2025:
    st.error("❌ 未找到任何年度数据！请先运行爬虫程序获取数据。")
    st.info("""
    **如何获取数据：**
    1. 运行 `python valorant_multi_year.py`
    2. 选择选项3（爬取2024+2025年数据）
    3. 等待数据爬取完成后重新加载此页面
    """)
    st.stop()

# ==================== 页面标题 ====================
st.markdown('<h1 class="main-title">🎮 VALORANT Champions 多年度对比分析</h1>', unsafe_allow_html=True)

year_badges = ""
if has_2024:
    year_badges += '<span class="year-badge year-2024">2024 Champions</span>'
if has_2025:
    year_badges += '<span class="year-badge year-2025">2025 Champions</span>'

st.markdown(f'<div style="text-align: center; margin-bottom: 2rem;">{year_badges}</div>', unsafe_allow_html=True)

# ==================== 顶部数据对比卡片 ====================
if has_both:
    st.markdown("### 📊 赛事规模对比")
    
    col1, col2, col3, col4 = st.columns(4)
    
    stats_2024 = {
        'matches': data_2024['match_id'].nunique() if 'match_id' in data_2024.columns else 0,
        'players': data_2024['player_name'].nunique(),
        'records': len(data_2024),
        'avg_rating': data_2024['Rating'].mean()
    }
    
    stats_2025 = {
        'matches': data_2025['match_id'].nunique() if 'match_id' in data_2025.columns else 0,
        'players': data_2025['player_name'].nunique(),
        'records': len(data_2025),
        'avg_rating': data_2025['Rating'].mean()
    }
    
    with col1:
        delta_matches = stats_2025['matches'] - stats_2024['matches']
        st.metric(
            "比赛场次", 
            f"2025: {stats_2025['matches']}", 
            f"{delta_matches:+d} vs 2024",
            delta_color="normal"
        )
        st.caption(f"2024: {stats_2024['matches']}场")
    
    with col2:
        delta_players = stats_2025['players'] - stats_2024['players']
        st.metric(
            "参赛选手", 
            f"2025: {stats_2025['players']}", 
            f"{delta_players:+d} vs 2024"
        )
        st.caption(f"2024: {stats_2024['players']}名")
    
    with col3:
        delta_records = stats_2025['records'] - stats_2024['records']
        st.metric(
            "数据记录", 
            f"2025: {stats_2025['records']}", 
            f"{delta_records:+d} vs 2024"
        )
        st.caption(f"2024: {stats_2024['records']}条")
    
    with col4:
        delta_rating = stats_2025['avg_rating'] - stats_2024['avg_rating']
        st.metric(
            "平均Rating", 
            f"2025: {stats_2025['avg_rating']:.3f}", 
            f"{delta_rating:+.3f} vs 2024"
        )
        st.caption(f"2024: {stats_2024['avg_rating']:.3f}")
    
    st.markdown("---")

# ==================== 侧边栏：年度选择和筛选器 ====================
with st.sidebar:
    st.title("🎯 分析控制面板")
    
    # 年度选择
    st.subheader("📅 选择分析年度")
    
    analysis_mode = st.radio(
        "分析模式",
        options=[
            "2024年单独分析" if has_2024 else None,
            "2025年单独分析" if has_2025 else None,
            "2024 vs 2025 对比" if has_both else None
        ],
        index=2 if has_both else (1 if has_2025 else 0)
    )
    
    st.markdown("---")
    
    # 根据选择模式设置当前数据
    if "2024年单独" in analysis_mode:
        current_data = data_2024
        current_year = "2024"
    elif "2025年单独" in analysis_mode:
        current_data = data_2025
        current_year = "2025"
    else:  # 对比模式
        current_data = pd.concat([data_2024, data_2025]) if has_both else (data_2025 if has_2025 else data_2024)
        current_year = "对比"
    
    # 筛选器
    st.subheader("🔍 数据筛选")
    
    min_rating = st.slider("最低Rating", 0.0, 2.0, 0.0, 0.1)
    min_appearances = st.slider("最小出场次数", 1, 20, 3)
    
    st.markdown("---")
    st.info(f"📌 当前分析模式：**{analysis_mode}**")

# ==================== 主要分析区域 ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 年度趋势对比",
    "⭐ 选手表现对比", 
    "🎭 英雄生态变化",
    "🗺️ 地图数据对比",
    "🏆 战队实力对比",
    "🔬 深度数据洞察"
])

# ==================== Tab 1: 年度趋势对比 ====================
with tab1:
    st.markdown('<h2 class="sub-title">📈 年度趋势对比分析</h2>', unsafe_allow_html=True)
    
    if has_both:
        # 1. 核心指标对比
        st.subheader("🎯 核心指标年度对比")
        
        # 分组显示不同量级的指标
        st.markdown("**方法1：按变化百分比对比（推荐）**")
        st.caption("通过百分比变化消除量级差异，直观反映趋势")
        
        metrics_to_compare = ['Rating', 'ACS', 'KAST', 'ADR', 'HS_Percent', 'KDA']
        
        # 计算两年的聚合数据
        agg_2024 = data_2024.groupby('player_name')[metrics_to_compare].mean().reset_index()
        agg_2025 = data_2025.groupby('player_name')[metrics_to_compare].mean().reset_index()
        
        comparison_data = []
        for metric in metrics_to_compare:
            val_2024 = agg_2024[metric].mean()
            val_2025 = agg_2025[metric].mean()
            comparison_data.append({
                'Metric': metric,
                '2024': val_2024,
                '2025': val_2025,
                'Change %': ((val_2025 - val_2024) / val_2024 * 100)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 可视化1：变化百分比柱状图（推荐）
        fig_change = go.Figure()
        
        colors = ['#FF4655' if x > 0 else '#667eea' for x in comparison_df['Change %']]
        
        fig_change.add_trace(go.Bar(
            x=comparison_df['Metric'],
            y=comparison_df['Change %'],
            marker_color=colors,
            text=comparison_df['Change %'].round(2),
            texttemplate='%{text:+.2f}%',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>变化: %{y:+.2f}%<extra></extra>'
        ))
        
        fig_change.update_layout(
            title="核心指标年度变化百分比",
            xaxis_title="<b><b>指标</b></b>",
            yaxis_title="<b><b>变化百分比 (%)</b></b>",
            height=500,
            hovermode='x unified',
            showlegend=False
        )
        
        # 添加参考线（0%基准线）
        fig_change.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig_change, use_container_width=True)
        
        # 可视化2：分组对比（按量级分组）
        st.markdown("---")
        st.markdown("**方法2：分组对比（按指标量级）**")
        st.caption("将不同量级的指标分组显示，避免视觉混淆")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**小数值指标**")
            small_metrics = ['Rating', 'KDA']
            
            fig_small = go.Figure()
            for metric in small_metrics:
                row = comparison_df[comparison_df['Metric'] == metric].iloc[0]
                fig_small.add_trace(go.Bar(
                    name=metric,
                    x=['2024', '2025'],
                    y=[row['2024'], row['2025']],
                    text=[f"{row['2024']:.2f}", f"{row['2025']:.2f}"],
                    textposition='outside'
                ))
            
            fig_small.update_layout(
                title="Rating / KDA",
                height=400,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(range=[0, 1.8]),  # 给KDA留出足够空间
                xaxis=dict(title=dict(text="<b>年份</b>")),
                yaxis_title=dict(text="<b>数值</b>")
            )
            st.plotly_chart(fig_small, use_container_width=True)
        
        with col2:
            st.markdown("**中数值指标**")
            medium_metrics = ['HS_Percent', 'KAST']
            
            fig_medium = go.Figure()
            for metric in medium_metrics:
                row = comparison_df[comparison_df['Metric'] == metric].iloc[0]
                fig_medium.add_trace(go.Bar(
                    name=metric,
                    x=['2024', '2025'],
                    y=[row['2024'], row['2025']],
                    text=[f"{row['2024']:.1f}", f"{row['2025']:.1f}"],
                    textposition='outside'
                ))
            
            fig_medium.update_layout(
                title="HS% / KAST",
                height=400,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(range=[0, 85]),  # 给KAST留出足够空间（约70-76）
                xaxis=dict(title=dict(text="<b>年份</b>")),
                yaxis_title=dict(text="<b>数值</b>")
            )
            st.plotly_chart(fig_medium, use_container_width=True)
        
        with col3:
            st.markdown("**大数值指标**")
            large_metrics = ['ADR', 'ACS']
            
            fig_large = go.Figure()
            for metric in large_metrics:
                row = comparison_df[comparison_df['Metric'] == metric].iloc[0]
                fig_large.add_trace(go.Bar(
                    name=metric,
                    x=['2024', '2025'],
                    y=[row['2024'], row['2025']],
                    text=[f"{row['2024']:.1f}", f"{row['2025']:.1f}"],
                    textposition='outside'
                ))
            
            fig_large.update_layout(
                title="ADR / ACS",
                height=400,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(range=[0, 250]),  # 给ACS留出足够空间（约193-189）
                xaxis=dict(title=dict(text="<b>年份</b>")),
                yaxis_title=dict(text="<b>数值</b>")
            )
            st.plotly_chart(fig_large, use_container_width=True)
        
        # 显示变化百分比
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(
                comparison_df.style.background_gradient(
                    subset=['Change %'], 
                    cmap='RdYlGn',
                    vmin=-20,
                    vmax=20
                ),
                use_container_width=True
            )
        
        with col2:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**💡 关键发现：**")
            
            # 自动生成洞察
            max_increase = comparison_df.loc[comparison_df['Change %'].idxmax()]
            max_decrease = comparison_df.loc[comparison_df['Change %'].idxmin()]
            
            if max_increase['Change %'] > 0:
                st.write(f"- ⬆️ **{max_increase['Metric']}** 增长最显著：**+{max_increase['Change %']:.1f}%**")
            
            if max_decrease['Change %'] < 0:
                st.write(f"- ⬇️ **{max_decrease['Metric']}** 下降最明显：**{max_decrease['Change %']:.1f}%**")
            
            # 平均Rating对比
            avg_rating_change = comparison_df[comparison_df['Metric'] == 'Rating']['Change %'].values[0]
            if avg_rating_change > 0:
                st.write(f"- 📊 2025年整体竞技水平提升 **{avg_rating_change:.1f}%**")
            else:
                st.write(f"- 📊 2025年整体竞技水平下降 **{abs(avg_rating_change):.1f}%**")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 2. 数据分布对比
        st.subheader("📊 数据分布变化趋势")
        
        metric_for_dist = st.selectbox(
            "选择指标查看分布变化",
            options=metrics_to_compare,
            format_func=lambda x: {
                'Rating': 'Rating (综合评分)',
                'ACS': 'ACS (战斗得分)',
                'KAST': 'KAST (参与率)',
                'ADR': 'ADR (平均伤害)',
                'HS_Percent': 'HS% (爆头率)',
                'KDA': 'KDA (击杀助攻比)'
            }.get(x, x)
        )
        
        # 小提琴图对比
        fig_violin = go.Figure()
        
        fig_violin.add_trace(go.Violin(
            y=data_2024[metric_for_dist],
            name='2024年',
            box_visible=True,
            meanline_visible=True,
            fillcolor='#667eea',
            opacity=0.6,
            x0='2024'
        ))
        
        fig_violin.add_trace(go.Violin(
            y=data_2025[metric_for_dist],
            name='2025年',
            box_visible=True,
            meanline_visible=True,
            fillcolor='#FF4655',
            opacity=0.6,
            x0='2025'
        ))
        
        fig_violin.update_layout(
            title=f"{metric_for_dist} 分布对比 (小提琴图)",
            xaxis_title="<b><b>年份</b></b>",
            yaxis_title=f"<b>{metric_for_dist}</b>",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig_violin, use_container_width=True)
        
        # 统计检验
        from scipy.stats import mannwhitneyu
        
        stat, p_value = mannwhitneyu(
            data_2024[metric_for_dist].dropna(),
            data_2025[metric_for_dist].dropna()
        )
        
        st.markdown('<div class="comparison-highlight">', unsafe_allow_html=True)
        st.markdown(f"""
        **统计显著性检验 (Mann-Whitney U Test):**
        - p-value: **{p_value:.4f}**
        - 结论: {'两年数据存在 **显著差异** (p < 0.05)' if p_value < 0.05 else '两年数据 **无显著差异** (p ≥ 0.05)'}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.info("⚠️ 需要2024和2025两年的数据才能进行对比分析。")

# ==================== Tab 2: 选手表现对比 ====================
with tab2:
    st.markdown('<h2 class="sub-title">⭐ 选手表现对比</h2>', unsafe_allow_html=True)
    
    if has_both:
        # 计算综合得分
        st.subheader("📊 选手综合能力评分体系")
        st.caption("基于多维指标加权计算综合得分：Rating(30%) + ACS(25%) + ADR(20%) + KAST(15%) + HS%(10%)")
        
        def calculate_comprehensive_score(df):
            """计算选手综合得分（标准化后加权）"""
            # 先按选手聚合
            agg_df = df.groupby('player_name').agg({
                'Rating': 'mean',
                'ACS': 'mean',
                'KAST': 'mean',
                'ADR': 'mean',
                'HS_Percent': 'mean',
                'KDA': 'mean',
                'match_id': 'count'
            }).rename(columns={'match_id': 'Games'}).reset_index()
            
            # 标准化（0-100）
            for col in ['Rating', 'ACS', 'KAST', 'ADR', 'HS_Percent']:
                if col in agg_df.columns:
                    min_val = agg_df[col].min()
                    max_val = agg_df[col].max()
                    if max_val > min_val:
                        agg_df[f'{col}_norm'] = (agg_df[col] - min_val) / (max_val - min_val) * 100
                    else:
                        agg_df[f'{col}_norm'] = 50
            
            # 综合得分（加权）
            agg_df['Comprehensive_Score'] = (
                agg_df['Rating_norm'] * 0.30 +
                agg_df['ACS_norm'] * 0.25 +
                agg_df['ADR_norm'] * 0.20 +
                agg_df['KAST_norm'] * 0.15 +
                agg_df['HS_Percent_norm'] * 0.10
            )
            
            return agg_df
        
        player_agg_2024 = calculate_comprehensive_score(data_2024)
        player_agg_2025 = calculate_comprehensive_score(data_2025)
        
        # TOP 10 综合实力对比
        st.markdown("---")
        st.subheader("🏆 TOP 10 选手综合实力对比")
        
        top10_2024 = player_agg_2024.nlargest(10, 'Comprehensive_Score')
        top10_2025 = player_agg_2025.nlargest(10, 'Comprehensive_Score')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 2024年 TOP 10（综合得分）")
            fig_2024 = px.bar(
                top10_2024.sort_values('Comprehensive_Score'),
                x='Comprehensive_Score',
                y='player_name',
                orientation='h',
                color='Comprehensive_Score',
                color_continuous_scale=SCALE_2024,
                text='Comprehensive_Score',
                hover_data=['Rating', 'ACS', 'ADR', 'KAST', 'HS_Percent']
            )
            fig_2024.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_2024.update_layout(
                height=500,
                showlegend=False,
                xaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_2024, use_container_width=True)
        
        with col2:
            st.markdown("#### 2025年 TOP 10（综合得分）")
            fig_2025 = px.bar(
                top10_2025.sort_values('Comprehensive_Score'),
                x='Comprehensive_Score',
                y='player_name',
                orientation='h',
                color='Comprehensive_Score',
                color_continuous_scale='Reds',
                text='Comprehensive_Score',
                hover_data=['Rating', 'ACS', 'ADR', 'KAST', 'HS_Percent']
            )
            fig_2025.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_2025.update_layout(
                height=500,
                showlegend=False,
                xaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_2025, use_container_width=True)
        
        # 分维度TOP 10对比
        st.markdown("---")
        st.subheader("📈 分维度 TOP 10 对比")
        
        metric_choice = st.selectbox(
            "选择维度查看TOP 10",
            options=['Rating', 'ACS', 'ADR', 'HS_Percent', 'KAST', 'KDA'],
            format_func=lambda x: {
                'Rating': 'Rating (综合评分)',
                'ACS': 'ACS (战斗得分)',
                'ADR': 'ADR (平均伤害)',
                'HS_Percent': 'HS% (爆头率)',
                'KAST': 'KAST (参与率)',
                'KDA': 'KDA (击杀助攻比)'
            }.get(x, x)
        )
        
        top10_metric_2024 = player_agg_2024.nlargest(10, metric_choice)
        top10_metric_2025 = player_agg_2025.nlargest(10, metric_choice)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**2024年 {metric_choice} TOP 10**")
            fig_m2024 = px.bar(
                top10_metric_2024.sort_values(metric_choice),
                x=metric_choice,
                y='player_name',
                orientation='h',
                color=metric_choice,
                color_continuous_scale=SCALE_2024,
                text=metric_choice
            )
            fig_m2024.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_m2024.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig_m2024, use_container_width=True)
        
        with col2:
            st.markdown(f"**2025年 {metric_choice} TOP 10**")
            fig_m2025 = px.bar(
                top10_metric_2025.sort_values(metric_choice),
                x=metric_choice,
                y='player_name',
                orientation='h',
                color=metric_choice,
                color_continuous_scale=SCALE_2025,
                text=metric_choice
            )
            fig_m2025.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_m2025.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig_m2025, use_container_width=True)
        
        # 跨年选手对比
        st.markdown("---")
        st.subheader("🔄 跨年度选手进步/退步分析（基于综合得分）")
        st.caption("使用综合得分衡量选手整体实力变化，更全面的反映进步情况")
        
        # 找出两年都参赛的选手
        common_players = set(player_agg_2024['player_name']) & set(player_agg_2025['player_name'])
        
        if common_players:
            progress_data = []
            
            for player in common_players:
                score_2024 = player_agg_2024[player_agg_2024['player_name'] == player]['Comprehensive_Score'].values[0]
                score_2025 = player_agg_2025[player_agg_2025['player_name'] == player]['Comprehensive_Score'].values[0]
                rating_2024 = player_agg_2024[player_agg_2024['player_name'] == player]['Rating'].values[0]
                rating_2025 = player_agg_2025[player_agg_2025['player_name'] == player]['Rating'].values[0]
                
                progress_data.append({
                    'player_name': player,
                    'Score_2024': score_2024,
                    'Score_2025': score_2025,
                    'Change': score_2025 - score_2024,
                    'Change_Percent': (score_2025 - score_2024) / score_2024 * 100,
                    'Rating_2024': rating_2024,
                    'Rating_2025': rating_2025
                })
            
            progress_df = pd.DataFrame(progress_data).sort_values('Change', ascending=False)
            
            # 获取TOP5进步和TOP5退步选手
            top5_improvers = progress_df.head(5)
            top5_decliners = progress_df.tail(5)
            
            fig_progress = go.Figure()
            
            # 添加TOP5进步选手（绿色）
            fig_progress.add_trace(go.Scatter(
                x=top5_improvers['Score_2024'],
                y=top5_improvers['Score_2025'],
                mode='markers+text',
                text=top5_improvers['player_name'],
                textposition='top center',
                name='TOP5 进步',
                marker=dict(
                    size=abs(top5_improvers['Change']),  # 根据变化幅度调整大小
                    color='#22c55e',  # 绿色
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                hovertemplate='<b>%{text}</b><br>2024: %{x:.1f}<br>2025: %{y:.1f}<br>变化: +%{customdata:.1f}<extra></extra>',
                customdata=top5_improvers['Change']
            ))
            
            # 添加TOP5退步选手（红色）
            fig_progress.add_trace(go.Scatter(
                x=top5_decliners['Score_2024'],
                y=top5_decliners['Score_2025'],
                mode='markers+text',
                text=top5_decliners['player_name'],
                textposition='bottom center',
                name='TOP5 退步',
                marker=dict(
                    size=abs(top5_decliners['Change']),  # 根据变化幅度调整大小
                    color='#ef4444',  # 红色
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                hovertemplate='<b>%{text}</b><br>2024: %{x:.1f}<br>2025: %{y:.1f}<br>变化: %{customdata:.1f}<extra></extra>',
                customdata=top5_decliners['Change']
            ))
            
            # 添加y=x参考线
            all_scores = pd.concat([top5_improvers[['Score_2024', 'Score_2025']], 
                                   top5_decliners[['Score_2024', 'Score_2025']]])
            min_val = min(all_scores['Score_2024'].min(), all_scores['Score_2025'].min()) - 5
            max_val = max(all_scores['Score_2024'].max(), all_scores['Score_2025'].max()) + 5
            
            fig_progress.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(dash='dash', color='gray', width=1.5),
                name='无变化线',
                showlegend=True
            ))
            
            fig_progress.update_layout(
                title="选手综合得分年度变化 (TOP 5进步 vs TOP 5退步)",
                xaxis_title="<b>2024 综合得分</b>",
                yaxis_title="<b>2025 综合得分</b>",
                height=650,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"
                )
            )
            
            st.plotly_chart(fig_progress, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 进步最大的5名选手**")
                st.dataframe(
                    progress_df.head(5)[['player_name', 'Score_2024', 'Score_2025', 'Change', 'Change_Percent']].style.format({
                        'Score_2024': '{:.1f}',
                        'Score_2025': '{:.1f}',
                        'Change': '{:+.1f}',
                        'Change_Percent': '{:+.1f}%'
                    }),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("**📉 退步最大的5名选手**")
                st.dataframe(
                    progress_df.tail(5)[['player_name', 'Score_2024', 'Score_2025', 'Change', 'Change_Percent']].style.format({
                        'Score_2024': '{:.1f}',
                        'Score_2025': '{:.1f}',
                        'Change': '{:+.1f}',
                        'Change_Percent': '{:+.1f}%'
                    }),
                    use_container_width=True
                )
            
            # 选手多维能力雷达图（可选任意选手）
            st.markdown("---")
            st.subheader("🧬 选手多维能力雷达图")
            st.caption("通过雷达图对比选手在多维指标上的相对位置（基于百分位标准化）")

            radar_year = st.selectbox("选择年份", ["2024", "2025"], key="player_radar_year")
            if radar_year == "2024":
                df_radar = player_agg_2024.copy()
            else:
                df_radar = player_agg_2025.copy()

            if not df_radar.empty:
                player_for_radar = st.selectbox(
                    "选择选手",
                    sorted(df_radar['player_name'].unique()),
                    key="player_radar_name"
                )

                metrics = ['Rating', 'ACS', 'ADR', 'KAST', 'HS_Percent']

                # 使用百分位数标准化到0-100
                df_radar_pct = df_radar.set_index('player_name')
                pct_values = df_radar_pct[metrics].rank(pct=True) * 100

                if player_for_radar in pct_values.index:
                    values = pct_values.loc[player_for_radar].values.tolist()
                    labels = ['Rating', 'ACS', 'ADR', 'KAST', 'HS%']

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=labels,
                        fill='toself',
                        name=player_for_radar
                    ))
                    fig_radar.update_layout(
                        title=f"{radar_year} 年选手能力雷达图 - {player_for_radar}",
                        polar=dict(
                            radialaxis=dict(range=[0, 100], showticklabels=True)
                        ),
                        showlegend=False,
                        height=500
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
            
        else:
            st.warning("⚠️ 未找到两年都参赛的选手")
    
    else:
        st.info("⚠️ 需要2024和2025两年的数据才能进行选手对比。")

# ==================== Tab 3: 英雄生态变化 ====================
with tab3:
    st.markdown('<h2 class="sub-title">🎭 英雄生态变化分析</h2>', unsafe_allow_html=True)
    
    if has_both:
        st.subheader("📊 英雄使用率年度对比")
        
        # 计算每年的英雄使用率
        agent_usage_2024 = data_2024['agent'].value_counts().reset_index()
        agent_usage_2024.columns = ['agent', 'count_2024']
        agent_usage_2024['usage_rate_2024'] = agent_usage_2024['count_2024'] / len(data_2024) * 100
        
        agent_usage_2025 = data_2025['agent'].value_counts().reset_index()
        agent_usage_2025.columns = ['agent', 'count_2025']
        agent_usage_2025['usage_rate_2025'] = agent_usage_2025['count_2025'] / len(data_2025) * 100
        
        # 合并数据
        agent_comparison = pd.merge(agent_usage_2024, agent_usage_2025, on='agent', how='outer').fillna(0)
        agent_comparison['change'] = agent_comparison['usage_rate_2025'] - agent_comparison['usage_rate_2024']
        agent_comparison = agent_comparison.sort_values('change', ascending=False)
        
        # 可视化
        fig_agent = go.Figure()
        
        fig_agent.add_trace(go.Bar(
            name='2024年',
            x=agent_comparison['agent'],
            y=agent_comparison['usage_rate_2024'],
            marker_color=COLOR_2024
        ))
        
        fig_agent.add_trace(go.Bar(
            name='2025年',
            x=agent_comparison['agent'],
            y=agent_comparison['usage_rate_2025'],
            marker_color=COLOR_2025
        ))
        
        fig_agent.update_layout(
            title="英雄使用率对比",
            xaxis_title="<b>英雄</b>",
            yaxis_title="<b>使用率 (%)</b>",
            barmode='group',
            height=600,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_agent, use_container_width=True)
        
        # 显示变化最大的英雄
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔥 使用率上升最多的英雄**")
            rising_agents = agent_comparison.nlargest(5, 'change')
            st.dataframe(
                rising_agents[['agent', 'usage_rate_2024', 'usage_rate_2025', 'change']].style.format({
                    'usage_rate_2024': '{:.1f}%',
                    'usage_rate_2025': '{:.1f}%',
                    'change': '{:+.1f}%'
                }),
                use_container_width=True
            )
        
        with col2:
            st.markdown("**❄️ 使用率下降最多的英雄**")
            falling_agents = agent_comparison.nsmallest(5, 'change')
            st.dataframe(
                falling_agents[['agent', 'usage_rate_2024', 'usage_rate_2025', 'change']].style.format({
                    'usage_rate_2024': '{:.1f}%',
                    'usage_rate_2025': '{:.1f}%',
                    'change': '{:+.1f}%'
                }),
                use_container_width=True
            )
        
        # 英雄胜率对比
        st.markdown("---")
        st.subheader("🎯 英雄表现(Rating)对比")
        
        agent_performance_2024 = data_2024.groupby('agent')['Rating'].mean().reset_index()
        agent_performance_2024.columns = ['agent', 'avg_rating_2024']
        
        agent_performance_2025 = data_2025.groupby('agent')['Rating'].mean().reset_index()
        agent_performance_2025.columns = ['agent', 'avg_rating_2025']
        
        agent_perf_comp = pd.merge(agent_performance_2024, agent_performance_2025, on='agent', how='outer').fillna(0)
        agent_perf_comp['rating_change'] = agent_perf_comp['avg_rating_2025'] - agent_perf_comp['avg_rating_2024']
        
        # 散点图：使用率 vs Rating变化
        agent_full = pd.merge(agent_comparison, agent_perf_comp, on='agent')
        
        fig_scatter = px.scatter(
            agent_full,
            x='usage_rate_2025',
            y='rating_change',
            size='count_2025',
            color='rating_change',
            color_continuous_scale=DIVERGING_SCALE,
            text='agent',
            labels={
                'usage_rate_2025': '2025年使用率 (%)',
                'rating_change': 'Rating变化',
                'count_2025': '使用次数'
            },
            title="英雄使用率 vs Rating变化"
        )
        
        fig_scatter.update_traces(textposition='top center')
        fig_scatter.update_layout(height=600)
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # 新增：英雄ACS排名对比
        st.markdown("---")
        st.subheader("💥 英雄ACS变化对比（2024 vs 2025）")
        st.caption("通过散点图对比不同英雄在两年中的平均ACS变化，识别版本中获益或受损最大的英雄")
        
        # 计算每个英雄在两年的平均ACS
        agent_acs_2024 = data_2024.groupby('agent').agg({
            'ACS': 'mean',
            'player_name': 'count'
        }).rename(columns={'player_name': 'usage_count_2024', 'ACS': 'ACS_2024'}).reset_index()
        
        agent_acs_2025 = data_2025.groupby('agent').agg({
            'ACS': 'mean',
            'player_name': 'count'
        }).rename(columns={'player_name': 'usage_count_2025', 'ACS': 'ACS_2025'}).reset_index()
        
        # 合并两年的ACS数据
        agent_acs_merge = pd.merge(agent_acs_2024, agent_acs_2025, on='agent', how='outer').fillna(0)
        agent_acs_merge['avg_usage'] = (agent_acs_merge['usage_count_2024'] + agent_acs_merge['usage_count_2025']) / 2
        agent_acs_merge['acs_change'] = agent_acs_merge['ACS_2025'] - agent_acs_merge['ACS_2024']
        
        # 过滤掉使用次数极少的英雄，避免噪声（例如总使用不足3局）
        agent_acs_merge = agent_acs_merge[agent_acs_merge['avg_usage'] >= 3]
        
        if not agent_acs_merge.empty:
            fig_acs_change = px.scatter(
                agent_acs_merge,
                x='ACS_2024',
                y='ACS_2025',
                size='avg_usage',
                color='acs_change',
                color_continuous_scale=DIVERGING_SCALE,
                text='agent',
                labels={
                    'ACS_2024': '2024年平均ACS',
                    'ACS_2025': '2025年平均ACS',
                    'avg_usage': '平均使用场次',
                    'acs_change': 'ACS变化(2025-2024)'
                },
                title="英雄ACS变化散点图（2024 vs 2025）"
            )
            fig_acs_change.update_traces(textposition='top center')
            fig_acs_change.update_layout(
                height=600,
                legend_title_text="ACS变化",
                xaxis=dict(title="2024年平均ACS"),
                yaxis=dict(title="2025年平均ACS")
            )
            st.plotly_chart(fig_acs_change, use_container_width=True)

        # 地图-位置-英雄强度分析
        st.markdown("---")
        st.subheader("🗺️ 地图×位置的英雄强度分析")
        st.caption("按地图和特工位置（决斗/先锋/哨位/控制）分组，评估不同位置的强势英雄组合")

        # 特工 → 位置映射（如有新英雄未覆盖，将归入“其他”）
        agent_role_map = {
            # 决斗者
            'Jett': '决斗者', 'Reyna': '决斗者', 'Raze': '决斗者', 'Phoenix': '决斗者',
            'Yoru': '决斗者', 'Neon': '决斗者', 'Iso': '决斗者', 'Waylay': '决斗者',
            # 先锋（Initiator）
            'Sova': '先锋', 'Skye': '先锋', 'Fade': '先锋', 'Kayo': '先锋',
            'Breach': '先锋', 'Gekko': '先锋', 'Tejo': '先锋',
            # 哨位（Sentinel）
            'Killjoy': '哨位', 'Cypher': '哨位', 'Chamber': '哨位', 'Deadlock': '哨位', 'Sage': '哨位', 'Vyse': '哨位',
            # 控制者（Controller）
            'Brimstone': '控制', 'Viper': '控制', 'Omen': '控制', 'Astra': '控制',
            'Harbor': '控制', 'Clove': '控制'
        }

        # 给两年数据打上位置标签
        data_2024_role = data_2024.copy()
        data_2025_role = data_2025.copy()
        data_2024_role['role'] = data_2024_role['agent'].map(agent_role_map).fillna('其他')
        data_2025_role['role'] = data_2025_role['agent'].map(agent_role_map).fillna('其他')

        # 位置强度随地图变化（折线图），对比 2024 vs 2025
        metric_for_role = st.selectbox(
            "选择评价指标（位置强度）",
            options=['ACS', 'ADR', 'KAST'],
            format_func=lambda x: {
                'ACS': 'ACS (战斗得分)',
                'ADR': 'ADR (平均伤害)',
                'KAST': 'KAST (参与率)'
            }.get(x, x),
            key="role_metric_line"
        )

        role_map_stats_2024 = data_2024_role.groupby(['map_name', 'role']).agg({
            metric_for_role: 'mean'
        }).reset_index()
        role_map_stats_2025 = data_2025_role.groupby(['map_name', 'role']).agg({
            metric_for_role: 'mean'
        }).reset_index()

        # 只保留两年都有的地图（交集）
        maps_2024 = set(role_map_stats_2024['map_name'].dropna().unique())
        maps_2025 = set(role_map_stats_2025['map_name'].dropna().unique())
        common_maps = maps_2024 & maps_2025
        map_order = sorted(list(common_maps))

        # 过滤掉非公共地图
        role_map_stats_2024 = role_map_stats_2024[role_map_stats_2024['map_name'].isin(common_maps)]
        role_map_stats_2025 = role_map_stats_2025[role_map_stats_2025['map_name'].isin(common_maps)]

        roles_order = sorted(list(set(role_map_stats_2024['role'].unique()) | set(role_map_stats_2025['role'].unique())))

        if map_order and roles_order:
            fig_role_line = make_subplots(
                rows=1,
                cols=2,
                shared_yaxes=True,
                subplot_titles=[
                    f"2024年各位置平均{metric_for_role}",
                    f"2025年各位置平均{metric_for_role}"
                ]
            )

            for role_name in roles_order:
                df24_r = role_map_stats_2024[role_map_stats_2024['role'] == role_name].set_index('map_name').reindex(map_order)
                df25_r = role_map_stats_2025[role_map_stats_2025['role'] == role_name].set_index('map_name').reindex(map_order)

                fig_role_line.add_trace(
                    go.Scatter(
                        x=map_order,
                        y=df24_r[metric_for_role],
                        mode='lines+markers',
                        name=role_name,
                        legendgroup=role_name
                    ),
                    row=1,
                    col=1
                )

                fig_role_line.add_trace(
                    go.Scatter(
                        x=map_order,
                        y=df25_r[metric_for_role],
                        mode='lines+markers',
                        name=role_name,
                        legendgroup=role_name,
                        showlegend=False
                    ),
                    row=1,
                    col=2
                )

            fig_role_line.update_xaxes(
                title_text="地图",
                categoryorder="array",
                categoryarray=map_order,
                row=1,
                col=1
            )
            fig_role_line.update_xaxes(
                title_text="地图",
                categoryorder="array",
                categoryarray=map_order,
                row=1,
                col=2
            )
            fig_role_line.update_yaxes(title_text=metric_for_role, row=1, col=1)
            fig_role_line.update_layout(
                height=500,
                legend_title_text="位置类型",
                title_text=f"不同地图上各位置的平均{metric_for_role}（2024 vs 2025）"
            )
            fig_role_line.update_traces(line=dict(dash='dash'))
            st.plotly_chart(fig_role_line, use_container_width=True)

            # 控制位单独视图：对比控制位在不同地图上的强度变化
            st.markdown("**🎯 控制位随地图的强度变化（2024 vs 2025）**")
            control_stats = pd.concat([
                role_map_stats_2024.assign(year='2024'),
                role_map_stats_2025.assign(year='2025')
            ], ignore_index=True)
            control_stats = control_stats[control_stats['role'] == '控制']

            if not control_stats.empty:
                fig_control = px.line(
                    control_stats,
                    x='map_name',
                    y=metric_for_role,
                    color='year',
                    markers=True,
                    category_orders={
                        'map_name': map_order,
                        'year': ['2024', '2025']
                    },
                    labels={
                        'map_name': '地图',
                        'year': '年份',
                        metric_for_role: metric_for_role
                    },
                    title=f"控制位在不同地图的平均{metric_for_role}（2024 vs 2025）"
                )
                fig_control.update_layout(height=400)
                fig_control.update_traces(line=dict(dash='dash'))
                st.plotly_chart(fig_control, use_container_width=True)

        # 针对某一位置，查看单张地图上的强势英雄排名
        st.markdown("**📌 按位置细看英雄强度（单张图）**")

        map_options = ["全部地图"] + sorted(list(set(data_2024_role['map_name'].dropna().unique()) | set(data_2025_role['map_name'].dropna().unique())))
        all_roles = sorted(list(set(data_2024_role['role'].unique()) | set(data_2025_role['role'].unique())))

        year_for_agent = st.selectbox("选择年份", ["2024", "2025"], key="agent_role_year")
        role_for_agent = st.selectbox("选择位置", all_roles, key="agent_role_choice")
        map_for_agent = st.selectbox("选择地图", map_options, key="agent_role_map")

        df_year_role = data_2024_role if year_for_agent == "2024" else data_2025_role
        if map_for_agent == "全部地图":
            df_year_role = df_year_role[df_year_role['role'] == role_for_agent]
        else:
            df_year_role = df_year_role[(df_year_role['role'] == role_for_agent) & (df_year_role['map_name'] == map_for_agent)]

        metric_for_agent = st.selectbox(
            "选择英雄评价指标",
            options=['Rating', 'ACS', 'ADR', 'KAST'],
            format_func=lambda x: {
                'Rating': 'Rating (综合评分)',
                'ACS': 'ACS (战斗得分)',
                'ADR': 'ADR (平均伤害)',
                'KAST': 'KAST (参与率)'
            }.get(x, x),
            key="agent_metric_choice"
        )

        if not df_year_role.empty:
            agent_stats = df_year_role.groupby('agent').agg({
                metric_for_agent: 'mean',
                'match_id': 'nunique'
            }).rename(columns={metric_for_agent: 'metric_value', 'match_id': 'games'}).reset_index()
            agent_stats = agent_stats.sort_values('metric_value', ascending=False)

            fig_agent_role = px.bar(
                agent_stats.head(10).sort_values('metric_value'),
                x='metric_value',
                y='agent',
                orientation='h',
                color='metric_value',
                color_continuous_scale='Viridis',
                text='metric_value',
                labels={
                    'agent': '英雄',
                    'metric_value': metric_for_agent
                },
                title=f"{year_for_agent} 年 {map_for_agent} 上 {role_for_agent} 英雄的{metric_for_agent}排名"
            )
            fig_agent_role.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_agent_role.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_agent_role, use_container_width=True)

        # 按特工类型查看在不同地图的表现（多特工折线图）
        st.markdown("---")
        st.markdown("**📈 按特工类型查看地图表现（多特工折线图）**")

        agent_metric_for_type = st.selectbox(
            "选择评价指标（特工表现）",
            options=['ACS', 'ADR', 'KAST'],
            format_func=lambda x: {
                'ACS': 'ACS (战斗得分)',
                'ADR': 'ADR (平均伤害)',
                'KAST': 'KAST (参与率)'
            }.get(x, x),
            key="agent_type_metric_choice"
        )

        role_for_type = st.selectbox(
            "选择特工类型",
            options=['决斗者', '先锋', '哨位', '控制'],
            key="agent_type_role_choice"
        )

        year_for_type = st.selectbox(
            "选择年份（特工折线）",
            options=["2024", "2025"],
            key="agent_type_year_choice"
        )

        df_type_year = data_2024_role if year_for_type == "2024" else data_2025_role
        df_type_year = df_type_year[df_type_year['role'] == role_for_type].copy()

        # 只保留两年共有的地图，保证与前文地图折线图一致
        maps_2024_all = set(data_2024_role['map_name'].dropna().unique())
        maps_2025_all = set(data_2025_role['map_name'].dropna().unique())
        common_maps_all = sorted(list(maps_2024_all & maps_2025_all))
        df_type_year = df_type_year[df_type_year['map_name'].isin(common_maps_all)]

        if not df_type_year.empty and common_maps_all:
            agent_map_stats = df_type_year.groupby(['map_name', 'agent']).agg({
                agent_metric_for_type: 'mean'
            }).reset_index()

            fig_agent_type_lines = px.line(
                agent_map_stats,
                x='map_name',
                y=agent_metric_for_type,
                color='agent',
                markers=True,
                category_orders={'map_name': common_maps_all},
                labels={
                    'map_name': '地图',
                    'agent': '英雄',
                    agent_metric_for_type: agent_metric_for_type
                },
                title=f"{year_for_type} 年 {role_for_type} 英雄在不同地图的平均{agent_metric_for_type}"
            )
            fig_agent_type_lines.update_traces(line=dict(dash='dash'))
            fig_agent_type_lines.update_layout(height=500)
            st.plotly_chart(fig_agent_type_lines, use_container_width=True)

    else:
        st.info("⚠️ 需要2024和2025两年的数据才能进行英雄生态对比。")

# ==================== Tab 4: 地图数据对比 ====================
with tab4:
    st.markdown('<h2 class="sub-title">🗺️ 地图战术特性对比</h2>', unsafe_allow_html=True)
    
    if has_both:
        st.subheader("📍 地图统计数据对比")
        
        # 计算每张地图的平均数据
        map_stats_2024 = data_2024.groupby('map_name').agg({
            'Rating': 'mean',
            'ACS': 'mean',
            'ADR': 'mean',
            'KAST': 'mean',
            'match_id': 'count'
        }).rename(columns={'match_id': 'Games'}).reset_index()
        map_stats_2024['year'] = '2024'
        
        map_stats_2025 = data_2025.groupby('map_name').agg({
            'Rating': 'mean',
            'ACS': 'mean',
            'ADR': 'mean',
            'KAST': 'mean',
            'match_id': 'count'
        }).rename(columns={'match_id': 'Games'}).reset_index()
        map_stats_2025['year'] = '2025'
        
        map_combined = pd.concat([map_stats_2024, map_stats_2025])
        
        # 选择指标
        metric_map = st.selectbox(
            "选择指标",
            options=['Rating', 'ACS', 'ADR', 'KAST'],
            format_func=lambda x: {
                'Rating': 'Rating (综合评分)',
                'ACS': 'ACS (战斗得分)',
                'ADR': 'ADR (平均伤害)',
                'KAST': 'KAST (参与率)'
            }.get(x, x)
        )
        
        fig_map = px.bar(
            map_combined,
            x='map_name',
            y=metric_map,
            color='year',
            barmode='group',
            color_discrete_map={'2024': '#667eea', '2025': '#FF4655'},
            title=f"地图 {metric_map} 对比"
        )
        
        fig_map.update_layout(height=500)
        st.plotly_chart(fig_map, use_container_width=True)
        
        # 地图池变化
        st.markdown("---")
        st.subheader("🔄 地图池变化")
        
        maps_2024 = set(data_2024['map_name'].unique())
        maps_2025 = set(data_2025['map_name'].unique())
        
        new_maps = maps_2025 - maps_2024
        removed_maps = maps_2024 - maps_2025
        common_maps = maps_2024 & maps_2025
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("共同地图", len(common_maps))
            if common_maps:
                st.write(", ".join(sorted(common_maps)))
        
        with col2:
            st.metric("2025新增", len(new_maps), delta=len(new_maps), delta_color="normal")
            if new_maps:
                st.write(", ".join(sorted(new_maps)))
        
        with col3:
            st.metric("移除地图", len(removed_maps), delta=-len(removed_maps) if removed_maps else 0, delta_color="inverse")
            if removed_maps:
                st.write(", ".join(sorted(removed_maps)))
        
    else:
        st.info("⚠️ 需要2024和2025两年的数据才能进行地图对比。")

# ==================== Tab 5: 战队实力对比 ====================
with tab5:
    st.markdown('<h2 class="sub-title">🏆 战队实力对比</h2>', unsafe_allow_html=True)
    
    if has_both:
        st.subheader("🥇 战队综合实力排名对比（基于综合得分）")
        st.caption("采用加权综合得分：Rating(30%) + ACS(25%) + ADR(20%) + KAST(15%) + HS%(10%)")
        
        def calculate_team_score(df):
            """计算战队综合得分"""
            team_agg = df.groupby('team').agg({
                'Rating': 'mean',
                'ACS': 'mean',
                'ADR': 'mean',
                'KAST': 'mean',
                'HS_Percent': 'mean',
                'KDA': 'mean'
            }).reset_index()
            
            # 标准化
            for col in ['Rating', 'ACS', 'ADR', 'KAST', 'HS_Percent']:
                if col in team_agg.columns:
                    min_val = team_agg[col].min()
                    max_val = team_agg[col].max()
                    if max_val > min_val:
                        team_agg[f'{col}_norm'] = (team_agg[col] - min_val) / (max_val - min_val) * 100
                    else:
                        team_agg[f'{col}_norm'] = 50
            
            # 综合得分
            team_agg['Comprehensive_Score'] = (
                team_agg['Rating_norm'] * 0.30 +
                team_agg['ACS_norm'] * 0.25 +
                team_agg['ADR_norm'] * 0.20 +
                team_agg['KAST_norm'] * 0.15 +
                team_agg['HS_Percent_norm'] * 0.10
            )

            # 战队类型划分（基于选手综合表现特征）
            def classify_team(row):
                rating_score = row.get('Rating_norm', 0)
                acs_score = row.get('ACS_norm', 0)
                adr_score = row.get('ADR_norm', 0)
                kast_score = row.get('KAST_norm', 0)
                hs_score = row.get('HS_Percent_norm', 0)

                normalized_center = [rating_score, acs_score, adr_score, kast_score, hs_score]
                mean_val = np.mean(normalized_center) if np.mean(normalized_center) != 0 else 0
                if mean_val > 0:
                    balance_score = 100 - (np.std(normalized_center) / mean_val * 100)
                else:
                    balance_score = 0

                # 火力型战队：Rating、ACS、ADR均较高（优先判断，提高门槛）
                if rating_score > 70 and acs_score > 70 and adr_score > 70:
                    return '🔥火力型战队'
                # 团队型战队：KAST排名前5（约前31%）
                elif kast_score > 68.75:
                    return '👥团队型战队'
                # 稳健型战队：各项指标较为均衡
                elif balance_score > 50 and rating_score > 30:
                    return '🎯稳健型战队'
                # 其余归为潜力型战队
                else:
                    return '🌱潜力型战队'

            team_agg['Team_Type'] = team_agg.apply(classify_team, axis=1)
            
            return team_agg
        
        team_stats_2024 = calculate_team_score(data_2024).sort_values('Comprehensive_Score', ascending=False)
        team_stats_2025 = calculate_team_score(data_2025).sort_values('Comprehensive_Score', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### 2024年所有战队（综合得分）- 共{len(team_stats_2024)}支")
            fig_team_2024 = px.bar(
                team_stats_2024.sort_values('Comprehensive_Score'),
                x='Comprehensive_Score',
                y='team',
                orientation='h',
                color='Comprehensive_Score',
                color_continuous_scale=SCALE_2024,
                text='Comprehensive_Score',
                hover_data=['Rating', 'ACS', 'ADR', 'KAST', 'Team_Type']
            )
            fig_team_2024.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_team_2024.update_layout(height=max(600, len(team_stats_2024) * 35), showlegend=False)
            st.plotly_chart(fig_team_2024, use_container_width=True)
        
        with col2:
            st.markdown(f"#### 2025年所有战队（综合得分）- 共{len(team_stats_2025)}支")
            fig_team_2025 = px.bar(
                team_stats_2025.sort_values('Comprehensive_Score'),
                x='Comprehensive_Score',
                y='team',
                orientation='h',
                color='Comprehensive_Score',
                color_continuous_scale='Reds',
                text='Comprehensive_Score',
                hover_data=['Rating', 'ACS', 'ADR', 'KAST', 'Team_Type']
            )
            fig_team_2025.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_team_2025.update_layout(height=max(600, len(team_stats_2025) * 35), showlegend=False)
            st.plotly_chart(fig_team_2025, use_container_width=True)

        # 战队类型分布总结（所有战队）
        st.markdown("---")
        st.markdown("**战队类型分布（基于选手综合表现的自动划分）**")

        # 统计各类型战队数量
        type_counts_2024 = team_stats_2024['Team_Type'].value_counts()
        type_counts_2025 = team_stats_2025['Team_Type'].value_counts()

        col_summary_2024, col_summary_2025 = st.columns(2)

        with col_summary_2024:
            st.markdown(f"**2024年所有战队类型（共{len(team_stats_2024)}支）**")
            for t, cnt in type_counts_2024.items():
                st.write(f"- {t}: {cnt} 支战队")
                # 显示该类型的具体战队
                teams_of_type = team_stats_2024[team_stats_2024['Team_Type'] == t]['team'].tolist()
                st.caption(f"  → {', '.join(teams_of_type)}")

        with col_summary_2025:
            st.markdown(f"**2025年所有战队类型（共{len(team_stats_2025)}支）**")
            for t, cnt in type_counts_2025.items():
                st.write(f"- {t}: {cnt} 支战队")
                # 显示该类型的具体战队
                teams_of_type = team_stats_2025[team_stats_2025['Team_Type'] == t]['team'].tolist()
                st.caption(f"  → {', '.join(teams_of_type)}")

        # 类型分布对比图（用于论文截图）
        type_df = pd.DataFrame({
            'Team_Type': list(type_counts_2024.index) + list(type_counts_2025.index),
            'Year': ['2024'] * len(type_counts_2024) + ['2025'] * len(type_counts_2025),
            'Count': list(type_counts_2024.values) + list(type_counts_2025.values)
        })

        # 使用环形图展示两年战队类型分布
        fig_team_type = make_subplots(
            rows=1,
            cols=2,
            specs=[[{'type': 'domain'}, {'type': 'domain'}]],
            subplot_titles=["2024年战队类型分布", "2025年战队类型分布"]
        )

        fig_team_type.add_trace(
            go.Pie(
                labels=type_counts_2024.index,
                values=type_counts_2024.values,
                hole=0.4,
                name='2024年'
            ),
            1, 1
        )

        fig_team_type.add_trace(
            go.Pie(
                labels=type_counts_2025.index,
                values=type_counts_2025.values,
                hole=0.4,
                name='2025年'
            ),
            1, 2
        )

        fig_team_type.update_traces(textposition='inside', textinfo='percent+label')
        fig_team_type.update_layout(
            title_text="所有战队类型分布对比（环形图）",
            height=450,
            legend_title_text="战队类型"
        )
        st.plotly_chart(fig_team_type, use_container_width=True)

        # 战队多维能力雷达图（可选任意战队）
        st.markdown("---")
        st.subheader("🧭 战队多维能力雷达图")
        st.caption("基于标准化得分的五维雷达图，直观展示战队能力结构")

        team_radar_year = st.selectbox("选择年份查看战队雷达图", ["2024", "2025"], key="team_radar_year")
        if team_radar_year == "2024":
            df_team_radar = team_stats_2024.copy()
        else:
            df_team_radar = team_stats_2025.copy()

        if not df_team_radar.empty:
            team_for_radar = st.selectbox(
                "选择战队",
                df_team_radar['team'].tolist(),
                key="team_radar_name"
            )

            radar_metrics = ['Rating_norm', 'ACS_norm', 'ADR_norm', 'KAST_norm', 'HS_Percent_norm']
            radar_labels = ['Rating', 'ACS', 'ADR', 'KAST', 'HS%']

            row_team = df_team_radar[df_team_radar['team'] == team_for_radar]
            if not row_team.empty:
                values_team = [row_team.iloc[0][m] for m in radar_metrics]

                fig_team_radar = go.Figure()
                fig_team_radar.add_trace(go.Scatterpolar(
                    r=values_team,
                    theta=radar_labels,
                    fill='toself',
                    name=team_for_radar
                ))
                fig_team_radar.update_layout(
                    title=f"{team_radar_year} 年战队能力雷达图 - {team_for_radar}",
                    polar=dict(
                        radialaxis=dict(range=[0, 100], showticklabels=True)
                    ),
                    showlegend=False,
                    height=500
                )
                st.plotly_chart(fig_team_radar, use_container_width=True)

        # 潜力型战队阵容诊断示例：Team Liquid (2025)
        st.markdown("---")
        st.subheader("🌱 潜力型战队阵容诊断示例：Team Liquid (2025)")
        st.caption("通过雷达图和位置热力图识别潜力型战队的短板，为阵容优化提供参考")

        # 仅使用2025年的战队数据
        tl_team_name = "Team Liquid"
        tl_row = team_stats_2025[team_stats_2025['team'] == tl_team_name]

        if not tl_row.empty:
            radar_metrics = ['Rating_norm', 'ACS_norm', 'ADR_norm', 'KAST_norm', 'HS_Percent_norm']
            radar_labels = ['Rating', 'ACS', 'ADR', 'KAST', 'HS%']

            # 参考基线：2025年稳健型战队平均（若没有稳健型，则使用全部战队平均）
            stable_teams = team_stats_2025[team_stats_2025['Team_Type'] == '🎯稳健型战队']
            if not stable_teams.empty:
                benchmark = stable_teams[radar_metrics].mean()
                benchmark_name = '稳健型战队平均'
            else:
                benchmark = team_stats_2025[radar_metrics].mean()
                benchmark_name = '全部战队平均'

            tl_values = [tl_row.iloc[0][m] for m in radar_metrics]
            benchmark_values = [benchmark[m] for m in radar_metrics]

            fig_tl_radar = go.Figure()
            fig_tl_radar.add_trace(go.Scatterpolar(
                r=tl_values,
                theta=radar_labels,
                fill='toself',
                name=tl_team_name,
                line=dict(color=COLOR_2025)
            ))
            fig_tl_radar.add_trace(go.Scatterpolar(
                r=benchmark_values,
                theta=radar_labels,
                fill='toself',
                name=benchmark_name,
                line=dict(color=COLOR_NEUTRAL)
            ))
            fig_tl_radar.update_layout(
                title=f"{tl_team_name} vs {benchmark_name} 能力雷达图 (2025)",
                polar=dict(radialaxis=dict(range=[0, 100], showticklabels=True)),
                height=500
            )
            st.plotly_chart(fig_tl_radar, use_container_width=True)

            # 构建 Team Liquid 在不同位置上的指标热力图
            # 复用英雄-位置映射
            agent_role_map_tl = {
                'Jett': '决斗者', 'Reyna': '决斗者', 'Raze': '决斗者', 'Phoenix': '决斗者',
                'Yoru': '决斗者', 'Neon': '决斗者', 'Iso': '决斗者', 'Waylay': '决斗者',
                'Sova': '先锋', 'Skye': '先锋', 'Fade': '先锋', 'Kayo': '先锋',
                'Breach': '先锋', 'Gekko': '先锋', 'Tejo': '先锋',
                'Killjoy': '哨位', 'Cypher': '哨位', 'Chamber': '哨位', 'Deadlock': '哨位', 'Sage': '哨位', 'Vyse': '哨位',
                'Brimstone': '控制', 'Viper': '控制', 'Omen': '控制', 'Astra': '控制',
                'Harbor': '控制', 'Clove': '控制'
            }

            tl_players = data_2025[data_2025['team'] == tl_team_name].copy()
            if not tl_players.empty:
                tl_players['role'] = tl_players['agent'].map(agent_role_map_tl).fillna('其他')

                role_metrics = ['Rating', 'ACS', 'ADR', 'KAST']
                tl_role_stats = tl_players.groupby('role')[role_metrics].mean()
                tl_role_stats = tl_role_stats.reindex(sorted(tl_role_stats.index))

                fig_tl_heat = px.imshow(
                    tl_role_stats,
                    x=tl_role_stats.columns,
                    y=tl_role_stats.index,
                    color_continuous_scale=SCALE_2025,
                    labels={'x': '指标', 'y': '位置类型', 'color': '数值'},
                    text_auto='.2f',
                    title=f"{tl_team_name} 不同位置的核心指标热力图 (2025)"
                )
                fig_tl_heat.update_layout(height=450)
                st.plotly_chart(fig_tl_heat, use_container_width=True)

    else:
        st.info("⚠️ 需要2024和2025两年的数据才能进行战队对比。")


# ==================== Tab 6: 深度数据洞察 ====================
with tab6:
    st.markdown('<h2 class="sub-title">🔬 深度数据洞察</h2>', unsafe_allow_html=True)
    
    if has_both:
        # 1. 指标相关性网络分析
        st.subheader("📊 指标相关性网络分析")
        st.caption("通过Pearson相关系数揭示核心指标之间的关联关系，发现Meta变化趋势")
        
        metrics_corr = ['Rating', 'ACS', 'KAST', 'ADR', 'HS_Percent', 'KDA']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**2024年指标相关性**")
            corr_2024 = data_2024[metrics_corr].corr()
            
            fig_corr_2024 = px.imshow(
                corr_2024,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                labels=dict(color="相关系数"),
                aspect="auto"
            )
            fig_corr_2024.update_layout(height=450)
            st.plotly_chart(fig_corr_2024, use_container_width=True)
        
        with col2:
            st.markdown("**2025年指标相关性**")
            corr_2025 = data_2025[metrics_corr].corr()
            
            fig_corr_2025 = px.imshow(
                corr_2025,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                labels=dict(color="相关系数"),
                aspect="auto"
            )
            fig_corr_2025.update_layout(height=450)
            st.plotly_chart(fig_corr_2025, use_container_width=True)
        
        # 相关性变化分析
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**💡 关键发现：**")
        
        rating_kast_2024 = corr_2024.loc['Rating', 'KAST']
        rating_kast_2025 = corr_2025.loc['Rating', 'KAST']
        change_kast = rating_kast_2025 - rating_kast_2024
        
        st.write(f"- Rating与KAST相关性：2024: **{rating_kast_2024:.3f}** → 2025: **{rating_kast_2025:.3f}** ({change_kast:+.3f})")
        st.write(f"  {'  👥 团队配合重要性提升' if change_kast > 0 else '  🎯 个人能力重要性提升'}")
        
        acs_adr_2024 = corr_2024.loc['ACS', 'ADR']
        acs_adr_2025 = corr_2025.loc['ACS', 'ADR']
        st.write(f"- ACS与ADR保持高度相关：2024: **{acs_adr_2024:.3f}**, 2025: **{acs_adr_2025:.3f}**")
        st.write(f"  🎯 验证了战斗得分与伤害输出的一致性")
        
        hs_rating_2024 = corr_2024.loc['HS_Percent', 'Rating']
        hs_rating_2025 = corr_2025.loc['HS_Percent', 'Rating']
        st.write(f"- HS%与Rating呈弱相关：2024: **{hs_rating_2024:.3f}**, 2025: **{hs_rating_2025:.3f}**")
        st.write(f"  💡 爆头率并非决定性因素")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 2. 数据分布稳定性分析
        st.markdown("---")
        st.subheader("📦 数据分布稳定性分析")
        st.caption("通过箱线图和变异系数分析两年数据的离散程度和稳定性")
        
        metric_box = st.selectbox(
            "选择指标查看箱线图",
            options=['Rating', 'ACS', 'KAST', 'ADR', 'HS_Percent'],
            format_func=lambda x: {
                'Rating': 'Rating (综合评分)',
                'ACS': 'ACS (战斗得分)',
                'KAST': 'KAST (参与率)',
                'ADR': 'ADR (平均伤害)',
                'HS_Percent': 'HS% (爆头率)'
            }.get(x, x),
            key='boxplot'
        )
        
        fig_box = go.Figure()
        
        fig_box.add_trace(go.Box(
            y=data_2024[metric_box],
            name='2024',
            marker_color=COLOR_2024,
            boxmean='sd'  # 显示标准差
        ))
        
        fig_box.add_trace(go.Box(
            y=data_2025[metric_box],
            name='2025',
            marker_color=COLOR_2025,
            boxmean='sd'
        ))
        
        fig_box.update_layout(
            title=f"{metric_box} 分布对比（箱线图）",
            yaxis_title=metric_box,
            height=500
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
        
        # 统计摘要
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**2024年统计**")
            median_2024 = data_2024[metric_box].median()
            std_2024 = data_2024[metric_box].std()
            mean_2024 = data_2024[metric_box].mean()
            cv_2024 = std_2024 / mean_2024 if mean_2024 != 0 else 0
            
            st.write(f"- 中位数: **{median_2024:.3f}**")
            st.write(f"- 标准差: **{std_2024:.3f}**")
            st.write(f"- 变异系数: **{cv_2024:.3f}**")
        
        with col2:
            st.markdown("**2025年统计**")
            median_2025 = data_2025[metric_box].median()
            std_2025 = data_2025[metric_box].std()
            mean_2025 = data_2025[metric_box].mean()
            cv_2025 = std_2025 / mean_2025 if mean_2025 != 0 else 0
            
            st.write(f"- 中位数: **{median_2025:.3f}**")
            st.write(f"- 标准差: **{std_2025:.3f}**")
            st.write(f"- 变异系数: **{cv_2025:.3f}**")
        
        with col3:
            st.markdown("**变化分析**")
            cv_change = cv_2025 - cv_2024
            cv_change_pct = (cv_change / cv_2024 * 100) if cv_2024 != 0 else 0
            
            st.write(f"- CV变化: **{cv_change:+.3f}**")
            st.write(f"- 变化率: **{cv_change_pct:+.1f}%**")
            
            if cv_change < 0:
                st.success("🎯 数据更稳定")
            else:
                st.warning("⚠️ 离散度增加")
        
        st.markdown('<div class="comparison-highlight">', unsafe_allow_html=True)
        st.markdown(f"""
        **💡 稳定性解读：**
        - 2025年{metric_box}的变异系数{'**降低**' if cv_change < 0 else '**升高**'} {abs(cv_change_pct):.1f}%
        - {'  🎯 选手水平趋于均衡，竞争更加激烈' if cv_change < 0 else '  🔥 数据离散度增加，超级明星效应显著'}
        """, unsafe_allow_html=False)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 3. 数据质量报告
        st.markdown("---")
        st.subheader("🎯 数据质量报告")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 2024年数据概览")
            st.write(f"- 总记录数：{len(data_2024)}")
            st.write(f"- 唯一选手：{data_2024['player_name'].nunique()}")
            st.write(f"- 唯一战队：{data_2024['team'].nunique()}")
            st.write(f"- 唯一英雄：{data_2024['agent'].nunique()}")
            st.write(f"- 唯一地图：{data_2024['map_name'].nunique()}")
            st.write(f"- Rating范围：{data_2024['Rating'].min():.2f} - {data_2024['Rating'].max():.2f}")
        
        with col2:
            st.markdown("#### 2025年数据概览")
            st.write(f"- 总记录数：{len(data_2025)}")
            st.write(f"- 唯一选手：{data_2025['player_name'].nunique()}")
            st.write(f"- 唯一战队：{data_2025['team'].nunique()}")
            st.write(f"- 唯一英雄：{data_2025['agent'].nunique()}")
            st.write(f"- 唯一地图：{data_2025['map_name'].nunique()}")
            st.write(f"- Rating范围：{data_2025['Rating'].min():.2f} - {data_2025['Rating'].max():.2f}")
        
        # 4. 综合评价
        st.markdown("---")
        st.subheader("📊 综合评价")
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**💡 年度对比总结：**")
        
        # 数据规模对比
        data_growth = ((len(data_2025) - len(data_2024)) / len(data_2024) * 100)
        st.write(f"- 📈 数据规模{'增长' if data_growth > 0 else '减少'} **{abs(data_growth):.1f}%**")
        
        # 选手数量对比
        player_growth = ((data_2025['player_name'].nunique() - data_2024['player_name'].nunique()) / data_2024['player_name'].nunique() * 100)
        st.write(f"- 👥 参赛选手{'增加' if player_growth > 0 else '减少'} **{abs(player_growth):.1f}%**")
        
        # 平均Rating对比
        avg_rating_2024 = data_2024['Rating'].mean()
        avg_rating_2025 = data_2025['Rating'].mean()
        rating_change = ((avg_rating_2025 - avg_rating_2024) / avg_rating_2024 * 100)
        
        st.write(f"- ⭐ 平均Rating{'提升' if rating_change > 0 else '下降'} **{abs(rating_change):.1f}%** ({avg_rating_2024:.3f} → {avg_rating_2025:.3f})")
        
        # 竞争激烈程度
        rating_cv_2024 = data_2024['Rating'].std() / data_2024['Rating'].mean()
        rating_cv_2025 = data_2025['Rating'].std() / data_2025['Rating'].mean()
        cv_improvement = ((rating_cv_2024 - rating_cv_2025) / rating_cv_2024 * 100) if rating_cv_2024 != 0 else 0
        
        if cv_improvement > 0:
            st.write(f"- 🎯 竞争激烈程度提升 **{cv_improvement:.1f}%**（选手水平更均衡）")
        else:
            st.write(f"- 🔥 明星效应增强 **{abs(cv_improvement):.1f}%**（实力差距扩大）")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 5. 选手类型聚类分析
        st.markdown("---")
        st.subheader("🎯 选手类型聚类分析 (K-means)")
        st.caption("基于多维指标将选手分类为不同类型，揭示战术风格变化")
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            def perform_clustering(df, n_clusters=4, year_label=''):
                """K-means聚类分析 - 优化版本"""
                # 准备数据
                player_agg = df.groupby('player_name')[['Rating', 'ACS', 'KAST', 'ADR', 'HS_Percent']].mean()
                
                # 标准化（使用Z-score保留相对关系）
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(player_agg)
                
                # K-means聚类
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(scaled_data)
                
                # 聚类中心（反标准化）
                centers = scaler.inverse_transform(kmeans.cluster_centers_)
                centers_df = pd.DataFrame(centers, columns=['Rating', 'ACS', 'KAST', 'ADR', 'HS_Percent'])
                
                # 智能分类：根据聚类中心特征自动识别类型
                cluster_labels = {}
                
                for i in range(n_clusters):
                    center = centers_df.iloc[i]
                    
                    # 计算特征得分（标准化到0-100）
                    rating_score = (center['Rating'] - player_agg['Rating'].min()) / (player_agg['Rating'].max() - player_agg['Rating'].min()) * 100
                    acs_score = (center['ACS'] - player_agg['ACS'].min()) / (player_agg['ACS'].max() - player_agg['ACS'].min()) * 100
                    adr_score = (center['ADR'] - player_agg['ADR'].min()) / (player_agg['ADR'].max() - player_agg['ADR'].min()) * 100
                    kast_score = center['KAST']
                    
                    # 计算数据平衡度（变异系数的倒数）
                    normalized_center = [
                        rating_score,
                        acs_score,
                        kast_score,
                        adr_score,
                        center['HS_Percent']
                    ]
                    balance_score = 100 - (np.std(normalized_center) / np.mean(normalized_center) * 100)
                    
                    # 分类逻辑
                    # 火力型：Rating、ACS、ADR都很高（>70分位）
                    if rating_score > 70 and acs_score > 70 and adr_score > 70:
                        cluster_labels[i] = '🔥火力型'
                    # 团队型：KAST特别高（>75%），但ACS/ADR中等
                    elif kast_score > 75 and acs_score < 70:
                        cluster_labels[i] = '👥团队型'
                    # 稳健型：各项指标都比较平衡（balance_score高）
                    elif balance_score > 60 and rating_score > 40:
                        cluster_labels[i] = '🎯稳健型'
                    # 潜力型：某几项特别突出，但不是全面高或全面平衡
                    else:
                        cluster_labels[i] = '🌱潜力型'
                
                # 如果有重复标签，按中心点综合得分排序重新命名
                if len(set(cluster_labels.values())) < n_clusters:
                    # 重新分配：按Rating高低排序
                    centers_df['综合得分'] = (
                        centers_df['Rating'] * 0.3 +
                        centers_df['ACS'] / player_agg['ACS'].max() * 100 * 0.25 +
                        centers_df['ADR'] / player_agg['ADR'].max() * 100 * 0.2 +
                        centers_df['KAST'] / 100 * 100 * 0.15 +
                        centers_df['HS_Percent'] / 100 * 100 * 0.1
                    )
                    centers_df = centers_df.sort_values('综合得分', ascending=False)
                    
                    # 按得分高低分配
                    sorted_indices = centers_df.index.tolist()
                    cluster_labels = {
                        sorted_indices[0]: '🔥火力型',
                        sorted_indices[1]: '🎯稳健型',
                        sorted_indices[2]: '👥团队型',
                        sorted_indices[3]: '🌱潜力型'
                    }
                
                # 映射到选手
                player_agg['Cluster'] = clusters
                player_agg['Cluster_Name'] = player_agg['Cluster'].map(cluster_labels)
                player_agg = player_agg.reset_index()
                
                return player_agg, centers_df, cluster_labels
            
            player_clusters_2024, centers_2024, labels_2024 = perform_clustering(data_2024, year_label='2024')
            player_clusters_2025, centers_2025, labels_2025 = perform_clustering(data_2025, year_label='2025')
            
            # 每年使用独立的智能识别标签
            cluster_names_2024 = [labels_2024.get(i, f'类型{i}') for i in range(4)]
            cluster_names_2025 = [labels_2025.get(i, f'类型{i}') for i in range(4)]
            
            # 3D散点图
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**2024年选手类型分布**")
                
                fig_3d_2024 = go.Figure()
                
                colors = ['#FF4655', '#00D9FF', '#FFD700', '#9370DB']
                
                for cluster in range(4):
                    cluster_data = player_clusters_2024[player_clusters_2024['Cluster'] == cluster]
                    fig_3d_2024.add_trace(go.Scatter3d(
                        x=cluster_data['ACS'],
                        y=cluster_data['KAST'],
                        z=cluster_data['Rating'],
                        mode='markers',
                        name=cluster_names_2024[cluster],
                        marker=dict(size=4, color=colors[cluster], opacity=0.6)
                    ))
                
                fig_3d_2024.update_layout(
                    scene=dict(
                        xaxis_title='ACS',
                        yaxis_title='KAST',
                        zaxis_title='Rating'
                    ),
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig_3d_2024, use_container_width=True)
            
            with col2:
                st.markdown("**2025年选手类型分布**")
                
                fig_3d_2025 = go.Figure()
                
                for cluster in range(4):
                    cluster_data = player_clusters_2025[player_clusters_2025['Cluster'] == cluster]
                    fig_3d_2025.add_trace(go.Scatter3d(
                        x=cluster_data['ACS'],
                        y=cluster_data['KAST'],
                        z=cluster_data['Rating'],
                        mode='markers',
                        name=cluster_names_2025[cluster],
                        marker=dict(size=4, color=colors[cluster], opacity=0.6)
                    ))
                
                fig_3d_2025.update_layout(
                    scene=dict(
                        xaxis_title='ACS',
                        yaxis_title='KAST',
                        zaxis_title='Rating'
                    ),
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig_3d_2025, use_container_width=True)
            
            # 聚类统计（按类型名称分组统计）
            st.markdown("**📊 类型分布对比：**")
            
            # 统计每种类型的数量
            type_count_2024 = player_clusters_2024['Cluster_Name'].value_counts()
            type_count_2025 = player_clusters_2025['Cluster_Name'].value_counts()
            
            # 合并所有可能的类型
            all_types = ['🔥火力型', '👥团队型', '🎯稳健型', '🌱潜力型']
            
            col1, col2, col3, col4 = st.columns(4)
            cols = [col1, col2, col3, col4]
            
            for i, type_name in enumerate(all_types):
                with cols[i]:
                    count_2024 = type_count_2024.get(type_name, 0)
                    count_2025 = type_count_2025.get(type_name, 0)
                    pct_2024 = count_2024 / len(player_clusters_2024) * 100 if len(player_clusters_2024) > 0 else 0
                    pct_2025 = count_2025 / len(player_clusters_2025) * 100 if len(player_clusters_2025) > 0 else 0
                    
                    st.metric(
                        type_name,
                        f"{pct_2025:.1f}%",
                        f"{pct_2025 - pct_2024:+.1f}%"
                    )
                    st.caption(f"2024: {pct_2024:.1f}%")
            
            # 每种类型的代表选手表格
            st.markdown("---")
            st.markdown("**👥 各类型代表选手详情：**")
            
            # 为2024和2025年分别展示
            tab_cluster_2024, tab_cluster_2025 = st.tabs(["2024年选手分类", "2025年选手分类"])
            
            with tab_cluster_2024:
                for i, name in enumerate(cluster_names_2024):
                    st.markdown(f"### {name} ({player_clusters_2024[player_clusters_2024['Cluster'] == i].shape[0]}名选手)")
                    
                    # 获取该类型的选手
                    cluster_players = player_clusters_2024[player_clusters_2024['Cluster'] == i].copy()
                    
                    # 按综合得分排序
                    cluster_players['Comprehensive_Score'] = (
                        cluster_players['Rating'] / cluster_players['Rating'].max() * 30 +
                        cluster_players['ACS'] / cluster_players['ACS'].max() * 25 +
                        cluster_players['ADR'] / cluster_players['ADR'].max() * 20 +
                        cluster_players['KAST'] / 100 * 15 +
                        cluster_players['HS_Percent'] / 100 * 10
                    )
                    
                    cluster_players = cluster_players.sort_values('Comprehensive_Score', ascending=False)
                    
                    # 展示表格
                    display_df = cluster_players[['player_name', 'Rating', 'ACS', 'KAST', 'ADR', 'HS_Percent', 'Comprehensive_Score']].head(10)
                    display_df.columns = ['选手', 'Rating', 'ACS', 'KAST%', 'ADR', 'HS%', '综合得分']
                    
                    st.dataframe(
                        display_df.style.format({
                            'Rating': '{:.2f}',
                            'ACS': '{:.0f}',
                            'KAST%': '{:.1f}',
                            'ADR': '{:.1f}',
                            'HS%': '{:.1f}',
                            '综合得分': '{:.1f}'
                        }).background_gradient(subset=['综合得分'], cmap='RdYlGn'),
                        use_container_width=True
                    )
                    
                    # 类型特征总结
                    avg_stats = cluster_players[['Rating', 'ACS', 'KAST', 'ADR', 'HS_Percent']].mean()
                    st.caption(f"📊 类型特征: Rating={avg_stats['Rating']:.2f} | ACS={avg_stats['ACS']:.0f} | KAST={avg_stats['KAST']:.1f}% | ADR={avg_stats['ADR']:.1f} | HS%={avg_stats['HS_Percent']:.1f}%")
                    st.markdown("")
            
            with tab_cluster_2025:
                for i, name in enumerate(cluster_names_2025):
                    st.markdown(f"### {name} ({player_clusters_2025[player_clusters_2025['Cluster'] == i].shape[0]}名选手)")
                    
                    # 获取该类型的选手
                    cluster_players = player_clusters_2025[player_clusters_2025['Cluster'] == i].copy()
                    
                    # 按综合得分排序
                    cluster_players['Comprehensive_Score'] = (
                        cluster_players['Rating'] / cluster_players['Rating'].max() * 30 +
                        cluster_players['ACS'] / cluster_players['ACS'].max() * 25 +
                        cluster_players['ADR'] / cluster_players['ADR'].max() * 20 +
                        cluster_players['KAST'] / 100 * 15 +
                        cluster_players['HS_Percent'] / 100 * 10
                    )
                    
                    cluster_players = cluster_players.sort_values('Comprehensive_Score', ascending=False)
                    
                    # 展示表格
                    display_df = cluster_players[['player_name', 'Rating', 'ACS', 'KAST', 'ADR', 'HS_Percent', 'Comprehensive_Score']].head(10)
                    display_df.columns = ['选手', 'Rating', 'ACS', 'KAST%', 'ADR', 'HS%', '综合得分']
                    
                    st.dataframe(
                        display_df.style.format({
                            'Rating': '{:.2f}',
                            'ACS': '{:.0f}',
                            'KAST%': '{:.1f}',
                            'ADR': '{:.1f}',
                            'HS%': '{:.1f}',
                            '综合得分': '{:.1f}'
                        }).background_gradient(subset=['综合得分'], cmap='RdYlGn'),
                        use_container_width=True
                    )
                    
                    # 类型特征总结
                    avg_stats = cluster_players[['Rating', 'ACS', 'KAST', 'ADR', 'HS_Percent']].mean()
                    st.caption(f"📊 类型特征: Rating={avg_stats['Rating']:.2f} | ACS={avg_stats['ACS']:.0f} | KAST={avg_stats['KAST']:.1f}% | ADR={avg_stats['ADR']:.1f} | HS%={avg_stats['HS_Percent']:.1f}%")
                    st.markdown("")
            
            # 聚类中心特征
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**💡 类型特征解读：**")
            
            st.write("""
            **🔥 火力型**：
            - 核心特征：**Rating、ACS、ADR均处于高位** (全部>70百分位)
            - 战术定位：进攻端火力输出，擅长制造击杀和伤害
            - 代表英雄：Jett, Raze, Reyna
            
            **👥 团队型**：
            - 核心特征：**KAST特别高** (>75%)，但ACS/ADR中等
            - 战术定位：团队配合核心，参团率高，存活能力强
            - 代表英雄：Omen, Viper, Killjoy
            
            **🎯 稳健型**：
            - 核心特征：**各项指标均衡**，无明显短板
            - 战术定位：全面型选手，适应性强，可靠稳定
            - 代表英雄：多英雄池
            
            **🌱 潜力型**：
            - 核心特征：**某几项特别突出**，但整体未达顶尖
            - 战术定位：有明显特长，有成长空间
            - 发展路径：可向火力型或稳健型转型
            """)
            
            # 分析最大变化（按类型名称统计）
            cluster_changes = []
            for type_name in all_types:
                count_2024 = type_count_2024.get(type_name, 0)
                count_2025 = type_count_2025.get(type_name, 0)
                pct_2024 = count_2024 / len(player_clusters_2024) * 100 if len(player_clusters_2024) > 0 else 0
                pct_2025 = count_2025 / len(player_clusters_2025) * 100 if len(player_clusters_2025) > 0 else 0
                cluster_changes.append((type_name, pct_2025 - pct_2024))
            
            cluster_changes.sort(key=lambda x: abs(x[1]), reverse=True)
            
            st.write(f"- {cluster_changes[0][0]} 占比{'**提升**' if cluster_changes[0][1] > 0 else '**下降**'} {abs(cluster_changes[0][1]):.1f}%")
            
            if cluster_changes[0][0] == '👥团队型' and cluster_changes[0][1] > 0:
                st.write("  💡 **Meta向团队配合方向演进**")
            elif cluster_changes[0][0] == '🔥火力型' and cluster_changes[0][1] > 0:
                st.write("  💡 **Meta向个人能力方向回归**")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except ImportError:
            st.warning("⚠️ 需要安装 scikit-learn 才能使用聚类分析功能：`pip install scikit-learn`")
        except Exception as e:
            st.error(f"聚类分析失败：{str(e)}")
        
        # 6. Rating预测模型
        st.markdown("---")
        st.subheader("🧠 Rating预测模型（多元回归）")
        st.caption("分析各指标对Rating的影响权重，揭示关键因素变化")
        
        try:
            from sklearn.linear_model import LinearRegression
            
            def build_rating_model(df, year_label=''):
                """Rating多元回归模型"""
                # 准备数据
                features = ['ACS', 'KAST', 'ADR', 'HS_Percent']
                X = df[features].dropna()
                y = df.loc[X.index, 'Rating']
                
                # 训练模型
                model = LinearRegression()
                model.fit(X, y)
                
                # 获取系数
                coefficients = dict(zip(features, model.coef_))
                intercept = model.intercept_
                score = model.score(X, y)
                
                return coefficients, intercept, score
            
            coef_2024, intercept_2024, r2_2024 = build_rating_model(data_2024, '2024')
            coef_2025, intercept_2025, r2_2025 = build_rating_model(data_2025, '2025')
            
            # 对比系数
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**2024年回归系数**")
                st.write(f"R² = **{r2_2024:.3f}**")
                
                for feature, coef in sorted(coef_2024.items(), key=lambda x: abs(x[1]), reverse=True):
                    st.write(f"- {feature}: **{coef:.4f}**")
            
            with col2:
                st.markdown("**2025年回归系数**")
                st.write(f"R² = **{r2_2025:.3f}**")
                
                for feature, coef in sorted(coef_2025.items(), key=lambda x: abs(x[1]), reverse=True):
                    st.write(f"- {feature}: **{coef:.4f}**")
            
            # 系数变化分析
            st.markdown("**📊 系数变化分析：**")
            
            coef_changes = {}
            for feature in coef_2024.keys():
                change = coef_2025[feature] - coef_2024[feature]
                change_pct = (change / abs(coef_2024[feature]) * 100) if coef_2024[feature] != 0 else 0
                coef_changes[feature] = (change, change_pct)
            
            # 排序（按绝对变化量）
            sorted_changes = sorted(coef_changes.items(), key=lambda x: abs(x[1][0]), reverse=True)
            
            fig_coef = go.Figure()
            
            features_list = [item[0] for item in sorted_changes]
            changes_list = [item[1][0] for item in sorted_changes]
            
            fig_coef.add_trace(go.Bar(
                x=features_list,
                y=changes_list,
                marker_color=['#FF4655' if x > 0 else '#667eea' for x in changes_list],
                text=[f"{x:+.4f}" for x in changes_list],
                textposition='outside'
            ))
            
            fig_coef.update_layout(
                title="回归系数变化（2024→ 2025）",
                xaxis_title="<b>指标</b>",
                yaxis_title="<b>系数变化量</b>",
                height=400,
                showlegend=False
            )
            
            fig_coef.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            st.plotly_chart(fig_coef, use_container_width=True)
            
            st.markdown('<div class="comparison-highlight">', unsafe_allow_html=True)
            st.markdown(f"""
            **💡 关键发现：**
            - **{sorted_changes[0][0]}** 对Rating的影响{'**增强**' if sorted_changes[0][1][0] > 0 else '**减弱**'} ({sorted_changes[0][1][1]:+.1f}%)
            - R²从 {r2_2024:.3f} 变化至 {r2_2025:.3f}{'  📈 模型解释能力提升' if r2_2025 > r2_2024 else '  📉 模型解释能力下降'}
            """, unsafe_allow_html=False)
            st.markdown('</div>', unsafe_allow_html=True)
            
        except ImportError:
            st.warning("⚠️ 需要安装 scikit-learn 才能使用预测模型：`pip install scikit-learn`")
        except Exception as e:
            st.error(f"模型构建失败：{str(e)}")
        
        # 7. 异常值检测
        st.markdown("---")
        st.subheader("🔍 异常表现检测 (Z-score)")
        st.caption("识别超常表现和数据异常，发现特殊战术时刻")
        
        metric_anomaly = st.selectbox(
            "选择指标检测异常值",
            options=['ACS', 'Rating', 'HS_Percent', 'ADR'],
            format_func=lambda x: {
                'ACS': 'ACS (战斗得分)',
                'Rating': 'Rating (综合评分)',
                'HS_Percent': 'HS% (爆头率)',
                'ADR': 'ADR (平均伤害)'
            }.get(x, x),
            key='anomaly'
        )
        
        # 计算Z-score
        def detect_anomalies(df, metric, threshold=2.5):
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            df['z_score'] = (df[metric] - mean_val) / std_val
            anomalies = df[abs(df['z_score']) > threshold].copy()
            return anomalies.sort_values('z_score', ascending=False)
        
        anomalies_2024 = detect_anomalies(data_2024.copy(), metric_anomaly)
        anomalies_2025 = detect_anomalies(data_2025.copy(), metric_anomaly)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**2024年 {metric_anomaly} 异常值 TOP 5**")
            if len(anomalies_2024) > 0:
                display_cols = ['player_name', 'team', 'map_name', metric_anomaly, 'z_score']
                st.dataframe(
                    anomalies_2024.head(5)[display_cols].style.format({
                        metric_anomaly: '{:.1f}',
                        'z_score': '{:.2f}'
                    }),
                    use_container_width=True
                )
            else:
                st.info("未检测到异常值")
        
        with col2:
            st.markdown(f"**2025年 {metric_anomaly} 异常值 TOP 5**")
            if len(anomalies_2025) > 0:
                display_cols = ['player_name', 'team', 'map_name', metric_anomaly, 'z_score']
                st.dataframe(
                    anomalies_2025.head(5)[display_cols].style.format({
                        metric_anomaly: '{:.1f}',
                        'z_score': '{:.2f}'
                    }),
                    use_container_width=True
                )
            else:
                st.info("未检测到异常值")
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(f"""
        **💡 异常检测说明：**
        - Z-score > 2.5 表示超常高表现，可能是：
          - 选手爆发状态
          - 特定战术完美执行
          - 对手失误造成机会
        - 2024年异常值数量：**{len(anomalies_2024)}**
        - 2025年异常值数量：**{len(anomalies_2025)}**
        """, unsafe_allow_html=False)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 8. 竞争激烈程度综合评估
        st.markdown("---")
        st.subheader("🏆 竞争激烈程度综合评估")
        
        st.markdown('<div class="comparison-highlight">', unsafe_allow_html=True)
        
        # 计算多维指标
        def calculate_competition_intensity(df, year_label):
            metrics = {}
            
            # 1. 数据离散度 (CV均值)
            cv_list = []
            for col in ['Rating', 'ACS', 'KAST', 'ADR']:
                cv = df[col].std() / df[col].mean()
                cv_list.append(cv)
            metrics['avg_cv'] = np.mean(cv_list)
            
            # 2. 顶尖选手占比
            player_scores = df.groupby('player_name')['Rating'].mean()
            top10_threshold = player_scores.quantile(0.9)
            metrics['top10_ratio'] = len(player_scores[player_scores >= top10_threshold]) / len(player_scores)
            
            # 3. 指标相关性强度
            corr_matrix = df[['Rating', 'ACS', 'KAST', 'ADR']].corr()
            metrics['avg_corr'] = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            
            # 4. 地图表现差异
            map_std_list = []
            for player in df['player_name'].unique():
                player_data = df[df['player_name'] == player]
                if len(player_data) > 2:
                    map_std_list.append(player_data['Rating'].std())
            metrics['avg_map_variance'] = np.mean(map_std_list) if len(map_std_list) > 0 else 0
            
            return metrics
        
        intensity_2024 = calculate_competition_intensity(data_2024, '2024')
        intensity_2025 = calculate_competition_intensity(data_2025, '2025')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cv_change = (intensity_2025['avg_cv'] - intensity_2024['avg_cv']) / intensity_2024['avg_cv'] * 100
            st.metric(
                "📊 数据离散度",
                f"{intensity_2025['avg_cv']:.3f}",
                f"{cv_change:+.1f}%"
            )
            st.caption("高=差距大")
        
        with col2:
            top10_change = (intensity_2025['top10_ratio'] - intensity_2024['top10_ratio']) * 100
            st.metric(
                "🎯 顶尖选手占比",
                f"{intensity_2025['top10_ratio']*100:.1f}%",
                f"{top10_change:+.1f}%"
            )
            st.caption("低=群英荟萃")
        
        with col3:
            corr_change = (intensity_2025['avg_corr'] - intensity_2024['avg_corr']) / intensity_2024['avg_corr'] * 100
            st.metric(
                "🔗 指标相关性",
                f"{intensity_2025['avg_corr']:.3f}",
                f"{corr_change:+.1f}%"
            )
            st.caption("高=全面型")
        
        with col4:
            variance_change = (intensity_2025['avg_map_variance'] - intensity_2024['avg_map_variance']) / intensity_2024['avg_map_variance'] * 100 if intensity_2024['avg_map_variance'] != 0 else 0
            st.metric(
                "🗺️ 地图表现差异",
                f"{intensity_2025['avg_map_variance']:.3f}",
                f"{variance_change:+.1f}%"
            )
            st.caption("低=更稳定")
        
        st.markdown("**💡 综合评估：**")
        
        # 智能分析
        insights = []
        
        if cv_change > 5:
            insights.append("• 选手实力差距**拉大**，明星效应显著")
        elif cv_change < -5:
            insights.append("• 选手水平**趋于均衡**，竞争更激烈")
        
        if top10_change > 5:
            insights.append("• 顶尖选手阶层**扩大**，高手更多")
        elif top10_change < -5:
            insights.append("• 顶尖选手**集中化**，精英主导")
        
        if corr_change > 5:
            insights.append("• 指标相关性增强，需要**全面发展**")
        elif corr_change < -5:
            insights.append("• 指标相关性降低，允许**风格化**")
        
        if variance_change < -5:
            insights.append("• 地图表现更稳定，**适应性提升**")
        elif variance_change > 5:
            insights.append("• 地图表现波动增大，**状态不稳**")
        
        if len(insights) > 0:
            for insight in insights:
                st.write(insight)
        else:
            st.write("• 两年竞争格局基本稳定，无明显变化")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.info("⚠️ 需要2024呔2025两年的数据才能查看深度洞察。")

# ==================== 页脚 ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>📊 数据来源: VLR.gg | 🎮 VALORANT Champions 2024 & 2025</p>
    <p>🔧 技术栈: Streamlit + Plotly + Pandas + SciPy + Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
