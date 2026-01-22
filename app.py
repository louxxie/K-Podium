"""
🏅 Beijing 2022 Olympics Analysis & 2026 Medal Prediction Dashboard
베이징 2022 동계올림픽 분석 및 2026 메달 예측 대시보드
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
from matplotlib import rc


# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 깃허브 리눅스 기준
if platform.system() == 'Linux':
    fontname = './NanumGothic.ttf'
    font_files = fm.findSystemFonts(fontpaths=fontname)
    fm.fontManager.addfont(fontname)
    fm._load_fontmanager(try_read_cache=False)
    rc('font', family='NanumGothic')


# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="🏅 Beijing 2022 Olympics Analysis",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 데이터 및 모델 로드 (캐싱)
# ============================================================
@st.cache_data
def load_data():
    """전처리된 데이터 로드"""
    try:
        df = pd.read_csv('beijing_data.csv')
        return df
    except FileNotFoundError:
        st.error("❌ 'beijing_data.csv' 파일을 찾을 수 없습니다. 먼저 노트북을 실행하여 데이터를 저장해주세요.")
        st.stop()

@st.cache_resource
def load_model():
    """학습된 모델 로드"""
    try:
        model_data = joblib.load('beijing_model.pkl')
        return model_data
    except FileNotFoundError:
        st.error("❌ 'beijing_model.pkl' 파일을 찾을 수 없습니다. 먼저 노트북을 실행하여 모델을 저장해주세요.")
        st.stop()

# 데이터 및 모델 로드
df = load_data()
model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']

# ============================================================
# 유틸리티 함수
# ============================================================
def classify_region(country):
    """국가를 지역별로 분류"""
    europe = ['Norway', 'Germany', 'Sweden', 'Netherlands', 'Austria', 'Switzerland', 
              'France', 'Italy', 'Slovenia', 'Finland', 'Great Britain', 'Hungary', 
              'Belgium', 'Czech Republic', 'Slovakia', 'Belarus', 'Spain', 'Ukraine', 
              'Estonia', 'Latvia', 'Poland', 'Roc']
    asia = ["People'S Republic Of China", 'Japan', 'Republic Of Korea']
    north_america = ['United States Of America', 'Canada']
    oceania = ['New Zealand', 'Australia']
    
    if country in europe:
        return '유럽'
    elif country in asia:
        return '아시아'
    elif country in north_america:
        return '북미'
    elif country in oceania:
        return '오세아니아'
    else:
        return '기타'

df['지역'] = df['국가명'].apply(classify_region)

# ============================================================
# 사이드바 메뉴
# ============================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Olympic_rings_without_rims.svg/1200px-Olympic_rings_without_rims.svg.png", 
                 use_container_width=True)
st.sidebar.title("🏅 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "페이지 선택",
    ["1. 대회 정보", "2. 대시 보드", "3. 모델 성능", "4. 메달 예측"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
**📊 Dashboard Info**
- 데이터: 베이징 2022 동계올림픽
- 참가국: 29개국
- 모델: 다중 회귀 분석
- 목적: 2026 메달 예측
""")

# ============================================================
# Page 1: 대회 정보
# ============================================================
if page == "1. 대회 정보":
    st.title("🏅 MILANO CORTINA 2026 Olympics")
    st.markdown("### 밀라노 코르티나 2026 동계올림픽 예측 현황 (베이징 2022 데이터 기반)")
    st.markdown("---")
    
    # KPI 카드
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📅 일정", "2026.2.6 ~ 2.22")
    
    with col2:
        st.metric("🏂 종목 개수", "16개")
    
    with col3:
        st.metric("🌍 참가국", "93개국")
    
    st.markdown("---")
    
    # 주요 일정 표시
    st.markdown("#### 🗓️ 주요 일정 및 관전 포인트")
    
    try:
        events_df = pd.read_csv('main_events.csv')
        st.dataframe(
            events_df,
            column_config={
                "날짜": st.column_config.TextColumn("날짜", width="small"),
                "종목": st.column_config.TextColumn("종목", width="small"),
                "주요 내용 및 기대 선수": st.column_config.TextColumn("주요 내용 및 기대 선수", width="large")
            },
            hide_index=True,
            use_container_width=True
        )
    except FileNotFoundError:
        st.warning("⚠️ 일정 파일(main_events.csv)을 찾을 수 없습니다.")

# ============================================================
# Page 2: 대시 보드
# ============================================================
elif page == "2. 대시 보드":
    st.title("🔍 Dashboard - 대시 보드")
    st.markdown("### 다양한 변수와 메달 수의 관계를 탐색해보세요")
    st.markdown("---")
    
    # 필터 섹션
    st.sidebar.markdown("### 🎯 필터 설정")
    
    # 메달 수 범위 필터
    medal_min, medal_max = st.sidebar.slider(
        "메달 수 범위",
        min_value=0,
        max_value=int(df['총메달'].max()),
        value=(0, int(df['총메달'].max())),
        step=1
    )
    
    # 지역 선택 필터
    regions = st.sidebar.multiselect(
        "지역 선택",
        options=df['지역'].unique().tolist(),
        default=df['지역'].unique().tolist()
    )
    
    # GDP 범위 필터
    gdp_min, gdp_max = st.sidebar.slider(
        "GDP 범위 (십억 USD)",
        min_value=float(df['GDP'].min()),
        max_value=float(df['GDP'].max()),
        value=(float(df['GDP'].min()), float(df['GDP'].max())),
        step=100.0
    )
    
    # 데이터 필터링
    filtered_df = df[
        (df['총메달'] >= medal_min) & 
        (df['총메달'] <= medal_max) &
        (df['지역'].isin(regions)) &
        (df['GDP'] >= gdp_min) &
        (df['GDP'] <= gdp_max)
    ].copy()
    
    # 필터링 결과 표시 (제거됨)
    # st.info(f"📊 필터링 결과: **{len(filtered_df)}개국** (전체 {len(df)}개국 중)")
    
    st.markdown("---")
    
    # 비교 변수 선택
    col_select, col_chart = st.columns([1, 3])
    
    with col_select:
        st.markdown("#### 📈 비교할 변수 선택")
        compare_var = st.selectbox(
            "변수 선택",
            options=['GDP', '인구수', '강설량', '기온', '행복지수', '인간개발지수', '올림픽선수단수'],
            index=0
        )
        
        # 상관계수 계산 (메달점수 기준)
        if len(filtered_df) > 1:
            correlation = filtered_df[compare_var].corr(filtered_df['메달점수'])
            st.metric("📊 상관계수", f"{correlation:.3f}")
            
            if abs(correlation) > 0.7:
                st.success("✅ 강한 상관관계")
            elif abs(correlation) > 0.4:
                st.info("ℹ️ 중간 상관관계")
            else:
                st.warning("⚠️ 약한 상관관계")
    
    with col_chart:
        st.markdown(f"#### 📊 {compare_var} vs 메달 점수")
        
        # 기본 산점도 (지역별 색상)
        fig_compare = px.scatter(
            filtered_df,
            x=compare_var,
            y='메달점수',
            size='메달점수',
            color='지역',
            hover_name='국가명',
            hover_data=['금메달', '은메달', '동메달', '메달점수'],
            labels={compare_var: compare_var, '메달점수': '메달 점수 (금10k/은1k/동100)'}
        )
        
        # 전체 추세선 추가 (지역 구분 없이)
        if len(filtered_df) > 1:
            fig_trend = px.scatter(
                filtered_df,
                x=compare_var,
                y='메달점수',
                trendline="ols"
            )
            # 추세선 트레이스 추출 및 추가
            if len(fig_trend.data) > 1:
                trendline_trace = fig_trend.data[1]
                trendline_trace.line.color = 'black'  # 검은색 실선 (아시아와 구별됨)
                trendline_trace.line.width = 3        # 두께 약간 증가
                trendline_trace.line.dash = 'dash'    # 점선 스타일
                trendline_trace.name = '전체 추세'
                fig_compare.add_trace(trendline_trace)
        
        fig_compare.update_layout(height=500)
        st.plotly_chart(fig_compare, use_container_width=True)
    
    st.markdown("---")


# ============================================================
# Page 3: 모델 성능
# ============================================================
elif page == "3. 모델 성능":
    st.title("📊 Model Performance - 모델 성능")
    st.markdown("### 다중 회귀 모델의 성능을 확인해보세요")
    st.markdown("---")
    
    # 모델 정보
    st.markdown("#### 🤖 모델 정보")
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.info("""
        **모델 타입**
        - 다중 선형 회귀 (Multiple Linear Regression)
        - 학습 데이터: 베이징 2022 (29개국)
        """)
    
    with col_info2:
        st.info(f"""
        **사용 특성 (Features)**
        - {', '.join(features)}
        """)
    
    with col_info3:
        st.info("""
        **예측 대상**
        - 메달 점수 (금:10,000 / 은:1,000 / 동:100)
        - 금메달 우선 원칙 반영
        """)
    
    st.markdown("---")
    
    # 테스트 세트 분할 (노트북과 동일하게)
    X = df[features]
    y = df['메달점수']  # 금:10000, 은:100, 동:1 가중치 적용
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 스케일링
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = scaler.transform(X_train)
    
    # 예측
    y_pred = model.predict(X_test_scaled)
    y_pred_train = model.predict(X_train_scaled)
    
    # 성능 지표 계산 (Train Set 기준)
    r2 = r2_score(y_train, y_pred_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae = mean_absolute_error(y_train, y_pred_train)
    
    # 성능 지표 표시
    st.markdown("#### 📈 모델 성능 지표")
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    
    with col_metric1:
        st.metric("R² Score (Train)", f"{r2:.4f}", 
                 help="결정계수: 1에 가까울수록 좋은 모델 (설명력)")
    
    with col_metric2:
        st.metric("RMSE (Train)", f"{rmse:.2f}점", 
                 help="평균 제곱근 오차: 낮을수록 좋음")
    
    with col_metric3:
        st.metric("MAE (Train)", f"{mae:.2f}점",
                 help="평균 절대 오차: 낮을수록 좋음")
    
    st.markdown("---")
    
    # 시각화 섹션
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.markdown("#### 🎯 예측값 vs 실제값")
        
        fig_pred = go.Figure()
        
        # 산점도
        fig_pred.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='예측값',
            marker=dict(size=10, color='blue', opacity=0.6),
            text=[f"실제: {actual:.1f}<br>예측: {pred:.1f}" for actual, pred in zip(y_test, y_pred)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # 이상적인 직선 (y=x)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig_pred.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='완벽한 예측선',
            line=dict(color='red', dash='dash')
        ))
        
        fig_pred.update_layout(
            xaxis_title='실제 메달 점수',
            yaxis_title='예측 메달 점수',
            height=400
        )
        st.plotly_chart(fig_pred, use_container_width=True)
    
    with col_viz2:
        st.markdown("#### 📊 잔차(Residual) 분포")
        
        residuals = y_test - y_pred
        
        fig_residual = px.histogram(
            x=residuals,
            nbins=20,
            labels={'x': '잔차 (실제 - 예측)', 'count': '빈도'},
            color_discrete_sequence=['skyblue']
        )
        fig_residual.add_vline(x=0, line_dash="dash", line_color="red")
        fig_residual.update_layout(height=400)
        st.plotly_chart(fig_residual, use_container_width=True)
    
    st.markdown("---")
    
    # 변수별 중요도
    st.markdown("#### 🔍 변수별 중요도 (회귀 계수)")
    
    feature_importance = pd.DataFrame({
        '변수': features,
        '계수': model.coef_,
        '절댓값': np.abs(model.coef_)
    }).sort_values('절댓값', ascending=False)
    
    fig_importance = px.bar(
        feature_importance,
        x='절댓값',
        y='변수',
        orientation='h',
        color='계수',
        color_continuous_scale=['red', 'white', 'blue'],
        labels={'절댓값': '중요도 (|계수|)', '변수': '특성'},
        text=feature_importance['계수'].round(3)
    )
    fig_importance.update_traces(textposition='outside')
    fig_importance.update_layout(height=400)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    st.markdown("---")
    
    # 국가별 예측 vs 실제 비교 테이블
    st.markdown("#### 📋 국가별 예측 vs 실제 비교 (Test Set)")
    
    # 테스트 세트의 국가명 가져오기
    test_countries = df.iloc[X_test.index]['국가명'].values
    
    comparison_df = pd.DataFrame({
        '국가명': test_countries,
        '실제 메달 점수': y_test.values,
        '예측 메달 점수': y_pred,
        '오차': y_test.values - y_pred,
        '오차율(%)': np.abs((y_test.values - y_pred) / y_test.values * 100)
    }).sort_values('실제 메달 점수', ascending=False)
    
    # 스타일링
    def highlight_error(row):
        if abs(row['오차']) < 3:
            return ['background-color: lightgreen'] * len(row)
        elif abs(row['오차']) < 5:
            return ['background-color: lightyellow'] * len(row)
        else:
            return ['background-color: lightcoral'] * len(row)
    
    st.dataframe(
        comparison_df.style.format({
            '실제 메달 점수': '{:.0f}',
            '예측 메달 점수': '{:.0f}',
            '오차': '{:.1f}',
            '오차율(%)': '{:.1f}'
        }),
        use_container_width=True,
        height=400
    )

# ============================================================
# Page 4: 메달 예측
# ============================================================
elif page == "4. 메달 예측":
    st.title("🔮 Medal Prediction - 2026 올림픽 메달 예측")
    st.markdown("### 국가 정보를 입력하여 2026년 예상 메달 수를 예측해보세요!")
    st.markdown("---")
    
    # 평균값 계산 (핵심 3가지 변수)
    avg_values = {
        'GDP': df['GDP'].mean(),
        '강설량': df['강설량'].mean(),
        '올림픽선수단수': df['올림픽선수단수'].mean()
    }
    
    # 예시 국가 데이터
    example_countries = {
        'Norway': df[df['국가명'] == 'Norway'].iloc[0],
        'Germany': df[df['국가명'] == 'Germany'].iloc[0],
        'Republic Of Korea': df[df['국가명'] == 'Republic Of Korea'].iloc[0]
    }
    
    # 입력 폼
    st.markdown("#### 📝 국가 정보 입력")
    
    # 예시 국가 버튼
    st.markdown("**🌍 예시 국가로 자동 입력:**")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🇳🇴 노르웨이", use_container_width=True):
            st.session_state.example = 'Norway'
    with col_btn2:
        if st.button("🇩🇪 독일", use_container_width=True):
            st.session_state.example = 'Germany'
    with col_btn3:
        if st.button("🇰🇷 한국", use_container_width=True):
            st.session_state.example = 'Republic Of Korea'
    
    st.markdown("---")
    
    # 입력 필드 (핵심 3가지 변수만)
    if 'example' in st.session_state:
        example = example_countries[st.session_state.example]
        default_gdp = float(example['GDP'])
        default_snow = int(example['강설량'])
        default_athletes = int(example['올림픽선수단수'])
    else:
        default_gdp = avg_values['GDP']
        default_snow = int(avg_values['강설량'])
        default_athletes = int(avg_values['올림픽선수단수'])
    
    gdp_input = st.number_input(
        f"💰 GDP (십억 USD) - 평균: {avg_values['GDP']:.2f}",
        min_value=0.0,
        max_value=30000.0,
        value=default_gdp,
        step=100.0,
        help="국가의 GDP를 입력하세요"
    )
    
    snowfall_input = st.number_input(
        f"❄️ 연평균 강설량 (cm) - 평균: {avg_values['강설량']:.0f}",
        min_value=0,
        max_value=1000,
        value=default_snow,
        step=1,
        help="국가의 연평균 강설량을 입력하세요"
    )
    
    athletes_input = st.number_input(
        f"🏃 올림픽 선수단 수 - 평균: {avg_values['올림픽선수단수']:.0f}",
        min_value=0,
        max_value=1000,
        value=default_athletes,
        step=1,
        help="파견할 선수단 규모를 입력하세요"
    )
    
    st.markdown("---")
    
    # 예측 버튼
    if st.button("🔮 메달 수 예측하기", type="primary", use_container_width=True):
        # 입력 데이터 준비 (핵심 3가지 변수만)
        input_data = np.array([[
            gdp_input,
            snowfall_input,
            athletes_input
        ]])
        
        # 스케일링
        input_scaled = scaler.transform(input_data)
        
        # 예측
        predicted_medals = model.predict(input_scaled)[0]
        
        # RMSE 기반 신뢰구간
        X_test_df = df[features].sample(frac=0.3, random_state=42)
        X_test_scaled = scaler.transform(X_test_df)
        y_test_df = df.loc[X_test_df.index, '메달점수']
        y_pred_test = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test_df, y_pred_test))
        
        # 메달 점수를 실제 메달 개수로 역계산
        # 가중치: 금 10,000 / 은 1,000 / 동 100
        gold_medals = int(predicted_medals // 10000)
        remaining = predicted_medals % 10000
        silver_medals = int(remaining // 1000)
        remaining2 = remaining % 1000
        bronze_medals = int(remaining2 // 100)
        total_medals = gold_medals + silver_medals + bronze_medals
        
        # 결과 표시
        st.success("### 🎯 예측 완료!")
        
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            st.metric("🏅 예측 메달 수", f"{total_medals}개",
                     help=f"금:{gold_medals}개 / 은:{silver_medals}개 / 동:{bronze_medals}개")
        
        with col_result2:
            st.metric("🥇 금메달", f"{gold_medals}개")
        
        # 메달 상세 정보
        col_medal1, col_medal2, col_medal3 = st.columns(3)
        
        with col_medal1:
            st.info(f"🥇 **금메달**: {gold_medals}개")
        
        with col_medal2:
            st.info(f"🥈 **은메달**: {silver_medals}개")
        
        with col_medal3:
            st.info(f"🥉 **동메달**: {bronze_medals}개")
        
        st.markdown("---")
        
# ============================================================
# 푸터
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🏅 Beijing 2022 Olympics Analysis Dashboard</p>
    <p>Powered by Streamlit & Plotly | Data Science Project 2024</p>
</div>
""", unsafe_allow_html=True)
