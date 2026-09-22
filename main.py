import streamlit as st
import pandas as pd

# ---------- 기본 설정 ----------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 뇌졸중 예측 실습실")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


df = load_data()

st.markdown("### 📌 이 데이터는 무엇일까요?")
st.write(
    "이 데이터는 여러 사람의 건강 정보와 생활 습관을 담고 있으며, "
    "그 사람이 뇌졸중을 겪었는지(1) 아닌지(0)를 함께 기록한 자료예요. "
    "아래에서 데이터를 하나씩 살펴볼까요?"
)

st.divider()

# ---------- 큰 숫자 카드 네 개 ----------
total_count = len(df)
col_count = df.shape[1]
stroke_count = int(df["stroke"].sum())
stroke_ratio = stroke_count / total_count * 100

card1, card2, card3, card4 = st.columns(4)
card1.metric("전체 사람 수", f"{total_count:,}명")
card2.metric("열 개수", f"{col_count}개")
card3.metric("뇌졸중 경험자 수", f"{stroke_count:,}명")
card4.metric("뇌졸중 비율", f"{stroke_ratio:.2f}%")

st.divider()

# ---------- 열 설명 표 ----------
st.markdown("### 📋 열 설명표")
st.caption("우리말 뜻 칸은 비어 있어요. 교재를 보면서 직접 채워 보세요!")

column_info = []
for col in df.columns:
    dtype = df[col].dtype
    if dtype == "object":
        value_kind = "문자(범주형)"
    else:
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1}):
            value_kind = "숫자(0 또는 1)"
        else:
            value_kind = "숫자"
    missing_count = df[col].isna().sum()

    column_info.append(
        {
            "열 이름": col,
            "우리말 뜻": "",
            "값의 종류": value_kind,
            "빈 값 개수": missing_count,
        }
    )

column_info_df = pd.DataFrame(column_info)

edited_column_info = st.data_editor(
    column_info_df,
    use_container_width=True,
    hide_index=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    key="column_info_editor",
)

st.divider()

# ---------- 데이터 처음 다섯 줄 ----------
st.markdown("### 🔍 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# ---------- 데이터 출처 ----------
st.markdown("### 📚 데이터 출처")
st.caption("교재에 적힌 출처 내용을 아래 칸에 옮겨 적어 보세요.")
st.text_area("출처를 입력하세요", value="", height=120, key="source_text")
