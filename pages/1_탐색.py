import streamlit as st
import pandas as pd
import plotly.express as px

# ---------- 기본 설정 ----------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 탐색",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 데이터 탐색")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


df = load_data()

# ---------- 1. 나이 & 평균 혈당 분포 히스토그램 ----------
st.markdown("### 1️⃣ 나이와 평균 혈당의 분포")

hist_col1, hist_col2 = st.columns(2)

with hist_col1:
    fig_age_hist = px.histogram(df, x="age", nbins=30, title="나이 분포")
    fig_age_hist.update_layout(xaxis_title="나이", yaxis_title="사람 수")
    st.plotly_chart(fig_age_hist, use_container_width=True)

with hist_col2:
    fig_glucose_hist = px.histogram(
        df, x="avg_glucose_level", nbins=30, title="평균 혈당 분포"
    )
    fig_glucose_hist.update_layout(xaxis_title="평균 혈당", yaxis_title="사람 수")
    st.plotly_chart(fig_glucose_hist, use_container_width=True)

st.divider()

# ---------- 2. 뇌졸중 여부에 따른 나이 & 평균 혈당 상자그림 ----------
st.markdown("### 2️⃣ 뇌졸중 여부에 따른 나이 · 평균 혈당 비교")

df_box = df.copy()
df_box["stroke_label"] = df_box["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

box_col1, box_col2 = st.columns(2)

with box_col1:
    fig_age_box = px.box(
        df_box, x="stroke_label", y="age", title="나이 비교",
        labels={"stroke_label": "뇌졸중 여부", "age": "나이"},
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with box_col2:
    fig_glucose_box = px.box(
        df_box, x="stroke_label", y="avg_glucose_level", title="평균 혈당 비교",
        labels={"stroke_label": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"},
    )
    st.plotly_chart(fig_glucose_box, use_container_width=True)

mean_table = (
    df_box.groupby("stroke_label")[["age", "avg_glucose_level"]]
    .mean()
    .round(2)
    .rename(columns={"age": "평균 나이", "avg_glucose_level": "평균 혈당"})
    .rename_axis("뇌졸중 여부")
    .reset_index()
)
st.dataframe(mean_table, use_container_width=True, hide_index=True)

st.divider()

# ---------- 3. 고혈압 / 심장병 유무에 따른 뇌졸중 비율 ----------
st.markdown("### 3️⃣ 고혈압 · 심장병 유무에 따른 뇌졸중 비율")


def stroke_ratio_by_group(data, group_col, group_label_map):
    grouped = (
        data.groupby(group_col)["stroke"]
        .mean()
        .mul(100)
        .round(2)
        .reset_index()
    )
    grouped[group_col] = grouped[group_col].map(group_label_map)
    grouped.columns = [group_col, "뇌졸중 비율(%)"]
    return grouped


bar_col1, bar_col2 = st.columns(2)

with bar_col1:
    hyper_group = stroke_ratio_by_group(
        df, "hypertension", {0: "고혈압 없음", 1: "고혈압 있음"}
    )
    fig_hyper = px.bar(
        hyper_group, x="hypertension", y="뇌졸중 비율(%)",
        title="고혈압 유무에 따른 뇌졸중 비율",
        labels={"hypertension": "고혈압 유무"},
    )
    st.plotly_chart(fig_hyper, use_container_width=True)

with bar_col2:
    heart_group = stroke_ratio_by_group(
        df, "heart_disease", {0: "심장병 없음", 1: "심장병 있음"}
    )
    fig_heart = px.bar(
        heart_group, x="heart_disease", y="뇌졸중 비율(%)",
        title="심장병 유무에 따른 뇌졸중 비율",
        labels={"heart_disease": "심장병 유무"},
    )
    st.plotly_chart(fig_heart, use_container_width=True)

st.divider()

# ---------- 4. bmi 결측치 뇌졸중 비율 vs 전체 뇌졸중 비율 ----------
st.markdown("### 4️⃣ 체질량지수(bmi) 결측 여부에 따른 뇌졸중 비율")

bmi_missing_df = df[df["bmi"].isna()]
bmi_missing_count = len(bmi_missing_df)
bmi_missing_stroke_ratio = (
    bmi_missing_df["stroke"].mean() * 100 if bmi_missing_count > 0 else 0
)
overall_stroke_ratio = df["stroke"].mean() * 100

bmi_compare_table = pd.DataFrame(
    {
        "구분": ["bmi 결측자", "전체"],
        "사람 수": [bmi_missing_count, len(df)],
        "뇌졸중 비율(%)": [
            round(bmi_missing_stroke_ratio, 2),
            round(overall_stroke_ratio, 2),
        ],
    }
)
st.dataframe(bmi_compare_table, use_container_width=True, hide_index=True)

st.divider()

# ---------- 5. 흡연 상태별 사람 수 ----------
st.markdown("### 5️⃣ 흡연 상태별 사람 수")

smoking_count_table = (
    df["smoking_status"]
    .value_counts()
    .reset_index()
)
smoking_count_table.columns = ["흡연 상태", "사람 수"]
st.dataframe(smoking_count_table, use_container_width=True, hide_index=True)
