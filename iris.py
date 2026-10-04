import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Iris Explorer",
    page_icon="🌸",
    layout="wide"
)

# --- Load Data ---
@st.cache_data
def load_data():
    df = sns.load_dataset('iris')
    return df

df_raw = load_data()
features = df_raw.columns[:-1].tolist()
species_list = df_raw['species'].unique().tolist()

# --- Sidebar ---
st.sidebar.title("🌸 Iris Explorer")
st.sidebar.markdown("Iris 데이터셋을 탐색하고 시각화하는 대시보드입니다.")

# 1. Species Filter (Checkboxes)
st.sidebar.subheader("종(Species) 선택")
selected_species = []
for s in species_list:
    if st.sidebar.checkbox(s, value=True):
        selected_species.append(s)

# 2. Features Filter (Checkboxes)
st.sidebar.subheader("표시할 특징 선택")
selected_features = []
for f in features:
    if st.sidebar.checkbox(f, value=True):
        selected_features.append(f)

# --- Filtering Logic ---
filtered_df = df_raw[df_raw['species'].isin(selected_species)]

# --- Error Handling ---
if not selected_species:
    st.error("최소 하나의 종(Species)을 선택해 주세요.")
    st.stop()

if not selected_features:
    st.error("최소 하나의 특징(Feature)을 선택해 주세요.")
    st.stop()

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["📋 데이터 개요", "📊 분포 분석", "🔗 관계 분석"])

# --- Tab 1: 데이터 개요 ---
with tab1:
    st.header("데이터 요약")
    
    # 상단 KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("선택된 샘플 수", f"{len(filtered_df)} / {len(df_raw)}")
    
    species_counts = filtered_df['species'].value_counts()
    
    # 종별 개수에 따른 색상 지정 함수
    def get_color(count):
        if count >= 50: return "green"
        elif count >= 30: return "orange"
        else: return "red"

    # KPI 스타일링 (HTML/CSS 사용)
    st.subheader("종별 분포")
    kpi_cols = st.columns(len(species_list))
    for i, s in enumerate(species_list):
        count = species_counts.get(s, 0)
        color = get_color(count)
        kpi_cols[i].markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
                <p style="margin: 0; font-size: 16px; color: #555;">{s.capitalize()}</p>
                <h2 style="margin: 0; color: {color};">{count}</h2>
            </div>
        """, unsafe_allow_html=True)

    col3.metric("선택된 특징 수", len(selected_features))

    # 기초 통계 테이블
    st.subheader("기초 통계")
    st.dataframe(filtered_df[selected_features].describe(), use_container_width=True)

    # 데이터 테이블
    st.divider()
    with st.expander("원본 데이터 보기"):
        st.subheader("필터링된 데이터")
        st.dataframe(filtered_df[selected_features + ['species']], use_container_width=True)

# --- Tab 2: 분포 분석 ---
with tab2:
    st.header("변수별 분포 분석")
    
    for col in selected_features:
        st.subheader(f"Feature: {col}")
        
        c1, c2 = st.columns(2)
        
        with c1:
            # Boxplot
            fig_box = px.box(filtered_df, x="species", y=col, color="species",
                             title=f"{col}의 종별 박스플롯")
            st.plotly_chart(fig_box, use_container_width=True)
            
        with c2:
            # Histogram
            fig_hist = px.histogram(filtered_df, x=col, color="species",
                                    marginal="rug", # 하단에 밀도 표시
                                    title=f"{col}의 종별 히스토그램",
                                    barmode="overlay")
            st.plotly_chart(fig_hist, use_container_width=True)

# --- Tab 3: 관계 분석 ---
with tab3:
    st.header("변수 간 관계 분석")
    
    # 1. Correlation Heatmap
    st.subheader("1. 상관계수 히트맵")
    if len(selected_features) > 1:
        corr = filtered_df[selected_features].corr()
        fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                             title="선택된 특징들 간의 상관관계")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("상관계수를 보려면 최소 2개 이상의 특징을 선택하세요.")

    st.divider()

    # 2. Scatter Plot (Interactive)
    st.subheader("2. 인터랙티브 산점도")
    col_x, col_y = st.columns(2)
    with col_x:
        x_axis = st.selectbox("X축 선택", options=selected_features, index=0)
    with col_y:
        # y축은 센스있게 x축과 다른 걸 기본값으로 (특징이 여러 개일 때)
        y_default_index = 1 if len(selected_features) > 1 else 0
        y_axis = st.selectbox("Y축 선택", options=selected_features, index=y_default_index)
        
    fig_scatter = px.scatter(filtered_df, x=x_axis, y=y_axis, color="species",
                             hover_data=selected_features,
                             title=f"{x_axis} vs {y_axis}")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # 3. Pairplot (Scatter Matrix)
    st.subheader("3. 페어플롯 (Scatter Matrix)")
    if len(selected_features) > 1:
        fig_pair = px.scatter_matrix(filtered_df, dimensions=selected_features, color="species",
                                     title="특징 간 관계 매트릭스",
                                     height=800)
        # 마커 크기 조절
        fig_pair.update_traces(diagonal_visible=False)
        st.plotly_chart(fig_pair, use_container_width=True)
    else:
        st.info("페어플롯을 보려면 최소 2개 이상의 특징을 선택하세요.")
