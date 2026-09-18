import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st


st.set_page_config(
    page_title="Flights 분석 대시보드",
    page_icon="✈️",
    layout="wide",
)


@st.cache_data
def load_flights() -> pd.DataFrame:
    return sns.load_dataset("flights")


def make_insights(filtered: pd.DataFrame) -> list[str]:
    if filtered.empty:
        return ["현재 필터 조건에 해당하는 데이터가 없습니다."]

    yearly = filtered.groupby("year", as_index=False)["passengers"].sum()
    monthly = filtered.groupby("month", as_index=False)["passengers"].mean()
    highest = filtered.loc[filtered["passengers"].idxmax()]
    lowest = filtered.loc[filtered["passengers"].idxmin()]
    insights = [
        f"선택된 기간의 총 승객 수는 {filtered['passengers'].sum():,}명이며, "
        f"월평균은 {filtered['passengers'].mean():,.0f}명입니다.",
        f"승객 수가 가장 많은 시점은 {int(highest['year'])}년 {highest['month']}로 "
        f"{int(highest['passengers']):,}명입니다.",
        f"승객 수가 가장 적은 시점은 {int(lowest['year'])}년 {lowest['month']}로 "
        f"{int(lowest['passengers']):,}명입니다.",
    ]

    if len(yearly) > 1 and yearly.iloc[0]["passengers"] != 0:
        growth = (yearly.iloc[-1]["passengers"] / yearly.iloc[0]["passengers"] - 1) * 100
        insights.append(
            f"선택 기간의 연간 승객 수는 시작 연도 대비 {growth:.1f}% "
            f"{('증가' if growth >= 0 else '감소')}했습니다."
        )

    busiest_month = monthly.loc[monthly["passengers"].idxmax(), "month"]
    quietest_month = monthly.loc[monthly["passengers"].idxmin(), "month"]
    insights.append(
        f"월평균 기준으로 {busiest_month}이 가장 붐비고, "
        f"{quietest_month}이 가장 한산합니다."
    )
    return insights


flights = load_flights()
min_year, max_year = int(flights["year"].min()), int(flights["year"].max())

st.title("✈️ 항공 승객 데이터 분석")
st.caption("Seaborn flights 데이터로 장기 성장 추세와 월별 계절성을 살펴봅니다.")

with st.sidebar:
    st.header("분석 조건")
    selected_years = st.slider(
        "분석 연도",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )
    selected_months = st.multiselect(
        "분석 월",
        options=flights["month"].drop_duplicates().tolist(),
        default=flights["month"].drop_duplicates().tolist(),
    )
    show_table = st.checkbox("상세 데이터 표시", value=True)

filtered = flights[
    flights["year"].between(selected_years[0], selected_years[1])
    & flights["month"].isin(selected_months)
].copy()

if filtered.empty:
    st.warning("선택한 조건에 맞는 데이터가 없습니다. 필터를 조정해 주세요.")
    st.stop()

yearly = filtered.groupby("year", as_index=False)["passengers"].sum()
monthly = (
    filtered.groupby("month", as_index=False)["passengers"]
    .mean()
    .rename(columns={"passengers": "average_passengers"})
)
month_order = flights["month"].drop_duplicates().tolist()
monthly["month"] = pd.Categorical(monthly["month"], categories=month_order, ordered=True)
monthly = monthly.sort_values("month")

first_year_total = yearly.iloc[0]["passengers"]
last_year_total = yearly.iloc[-1]["passengers"]
growth = ((last_year_total / first_year_total) - 1) * 100 if first_year_total else 0
peak = filtered.loc[filtered["passengers"].idxmax()]

metric_columns = st.columns(4)
metric_columns[0].metric("총 승객 수", f"{filtered['passengers'].sum():,.0f}명")
metric_columns[1].metric("월평균 승객 수", f"{filtered['passengers'].mean():,.0f}명")
metric_columns[2].metric(
    "선택 기간 성장률",
    f"{growth:+.1f}%",
    help="선택 범위의 첫 연도와 마지막 연도 연간 승객 수를 비교합니다.",
)
metric_columns[3].metric(
    "최고 승객 수",
    f"{peak['passengers']:,.0f}명",
    f"{int(peak['year'])}년 {peak['month']}",
)

st.subheader("주요 데이터 인사이트")
for insight in make_insights(filtered):
    st.markdown(f"- {insight}")

left_chart, right_chart = st.columns(2)
with left_chart:
    st.subheader("연도별 승객 수 추이")
    year_centered = yearly["year"] - yearly["year"].mean()
    passengers_centered = yearly["passengers"] - yearly["passengers"].mean()
    year_variance = (year_centered**2).sum()
    trend_slope = (
        (year_centered * passengers_centered).sum() / year_variance
        if year_variance
        else 0
    )
    trend_intercept = yearly["passengers"].mean() - trend_slope * yearly["year"].mean()
    yearly["trend"] = trend_slope * yearly["year"] + trend_intercept

    yearly_figure = px.line(
        yearly,
        x="year",
        y="passengers",
        markers=True,
        labels={"year": "연도", "passengers": "승객 수"},
    )
    yearly_figure.add_trace(
        go.Scatter(
            x=yearly["year"],
            y=yearly["trend"],
            mode="lines",
            name="직선 추세",
            line={"dash": "dash", "color": "#D95F02"},
        )
    )
    yearly_figure.update_layout(hovermode="x unified")
    st.plotly_chart(yearly_figure, width="stretch")

with right_chart:
    st.subheader("월별 평균 승객 수")
    monthly_figure = px.bar(
        monthly,
        x="month",
        y="average_passengers",
        labels={"month": "월", "average_passengers": "평균 승객 수"},
        color="average_passengers",
        color_continuous_scale="YlGnBu",
    )
    st.plotly_chart(monthly_figure, width="stretch")

st.subheader("월별·연도별 승객 수 히트맵")
pivot = filtered.pivot(index="month", columns="year", values="passengers")
pivot = pivot.reindex(month_order).dropna(how="all")
heatmap_figure = px.imshow(
    pivot,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="YlGnBu",
    labels={"x": "연도", "y": "월", "color": "승객 수"},
)
st.plotly_chart(heatmap_figure, width="stretch")

if show_table:
    st.subheader("상세 데이터")
    st.dataframe(filtered, width="stretch", hide_index=True)

    csv_buffer = io.StringIO()
    filtered.to_csv(csv_buffer, index=False)
    st.download_button(
        "필터링 데이터 CSV 다운로드",
        data=csv_buffer.getvalue(),
        file_name="flights_filtered.csv",
        mime="text/csv",
    )