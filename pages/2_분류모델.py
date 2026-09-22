import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

# ---------- 기본 설정 ----------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 분류 모델",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 분류 모델")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

FEATURE_LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
ALL_FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


df = load_data()

# ---------- 1. 속성 선택 ----------
st.markdown("### 1️⃣ 모델에 사용할 속성 고르기")

selected_raw = st.multiselect(
    "속성을 두 개 이상 골라 주세요.",
    options=ALL_FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda x: FEATURE_LABELS[x],
)
# 항상 같은 순서를 유지하도록 정렬
selected_features = [f for f in ALL_FEATURES if f in selected_raw]

if len(selected_features) < 2:
    st.warning("⚠️ 속성을 두 개 이상 골라야 모델을 만들 수 있어요.")
    st.stop()

st.caption("고른 속성: " + ", ".join(FEATURE_LABELS[f] for f in selected_features))

st.divider()

# ---------- 2. 데이터 준비 ----------
df_sorted = df.sort_values("id").reset_index(drop=True)
pos_in_group = df_sorted.index % 10
test_mask = pos_in_group < 3

test_df = df_sorted[test_mask].copy()
train_df = df_sorted[~test_mask].copy()

# bmi는 고른 경우에만, 훈련용의 중앙값으로 결측치를 채움
if "bmi" in selected_features:
    bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)

X_train = train_df[selected_features]
y_train = train_df["stroke"]
X_test = test_df[selected_features]
y_test = test_df["stroke"]

# 훈련용(7명)으로만 크기 맞추기(업샘플링)
train_majority = train_df[train_df["stroke"] == 0]
train_minority = train_df[train_df["stroke"] == 1]

if 0 < len(train_minority) < len(train_majority):
    train_minority_up = resample(
        train_minority,
        replace=True,
        n_samples=len(train_majority),
        random_state=42,
    )
    train_balanced = pd.concat([train_majority, train_minority_up])
else:
    train_balanced = train_df.copy()

train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

X_train_bal = train_balanced[selected_features]
y_train_bal = train_balanced["stroke"]

# ---------- 3. 모델 학습 ----------
model_lr = LogisticRegression(random_state=42, max_iter=1000)
model_lr.fit(X_train_bal, y_train_bal)

model_dt = DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=42)
model_dt.fit(X_train_bal, y_train_bal)


def get_accuracies(model):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return train_acc, test_acc


lr_train_acc, lr_test_acc = get_accuracies(model_lr)
dt_train_acc, dt_test_acc = get_accuracies(model_dt)

majority_class = y_train.mode()[0]
baseline_train_acc = accuracy_score(y_train, [majority_class] * len(y_train))
baseline_test_acc = accuracy_score(y_test, [majority_class] * len(y_test))

# ---------- 4. 정확도 카드 ----------
st.markdown("### 2️⃣ 모델 정확도 비교")

card1, card2, card3 = st.columns(3)

with card1:
    st.metric("로지스틱 회귀(확률로 답하는 모델)", f"{lr_test_acc * 100:.1f}%")
    st.caption(f"훈련 정확도 {lr_train_acc * 100:.1f}% · 테스트 정확도 {lr_test_acc * 100:.1f}%")

with card2:
    st.metric("의사결정트리(질문으로 답하는 모델)", f"{dt_test_acc * 100:.1f}%")
    st.caption(f"훈련 정확도 {dt_train_acc * 100:.1f}% · 테스트 정확도 {dt_test_acc * 100:.1f}%")

with card3:
    st.metric("기준 모델(입력을 보지 않고 다수 쪽으로만 답하는 모델)", f"{baseline_test_acc * 100:.1f}%")
    st.caption(
        f"훈련 정확도 {baseline_train_acc * 100:.1f}% · 테스트 정확도 {baseline_test_acc * 100:.1f}%"
    )

st.divider()

# ---------- 5. 산점도 + 분류 경계 ----------
st.markdown("### 3️⃣ 두 속성으로 살펴보는 분류 경계")

axis_col1, axis_col2 = st.columns(2)
with axis_col1:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        format_func=lambda x: FEATURE_LABELS[x],
        index=0,
        key="x_axis_feature",
    )
with axis_col2:
    y_options = [f for f in selected_features if f != x_feature]
    y_feature = st.selectbox(
        "세로축 속성",
        options=y_options,
        format_func=lambda x: FEATURE_LABELS[x],
        index=0,
        key="y_axis_feature",
    )

other_features = [f for f in selected_features if f not in (x_feature, y_feature)]
fixed_values = {f: float(test_df[f].median()) for f in other_features}

if fixed_values:
    fixed_text = ", ".join(f"{FEATURE_LABELS[f]} = {fixed_values[f]:.2f}" for f in other_features)
    st.caption(f"그림에 나타나지 않는 속성은 테스트 데이터의 중앙값으로 고정했어요: {fixed_text}")
else:
    st.caption("고른 속성이 두 개뿐이라 따로 고정할 속성이 없어요.")

# 그림 범위 계산
x_min, x_max = test_df[x_feature].min(), test_df[x_feature].max()
y_min, y_max = test_df[y_feature].min(), test_df[y_feature].max()
x_margin = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
y_margin = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
x_range = [x_min - x_margin, x_max + x_margin]
y_range = [y_min - y_margin, y_max + y_margin]

# 의사결정트리 영역을 칠하기 위한 격자
xx = np.linspace(x_range[0], x_range[1], 150)
yy = np.linspace(y_range[0], y_range[1], 150)
xx_grid, yy_grid = np.meshgrid(xx, yy)

grid_df = pd.DataFrame({x_feature: xx_grid.ravel(), y_feature: yy_grid.ravel()})
for f in other_features:
    grid_df[f] = fixed_values[f]
grid_df = grid_df[selected_features]

Z_dt = model_dt.predict(grid_df).reshape(xx_grid.shape)

# 로지스틱 회귀 경계선(확률 0.5) 계산
coef = model_lr.coef_[0]
intercept = model_lr.intercept_[0]
idx_x = selected_features.index(x_feature)
idx_y = selected_features.index(y_feature)
coef_x = coef[idx_x]
coef_y = coef[idx_y]
constant_term = intercept + sum(
    coef[selected_features.index(f)] * fixed_values[f] for f in other_features
)

line_x_vals, line_y_vals = [], []
line_note = ""

if abs(coef_y) > 1e-9:
    xs = np.linspace(x_range[0], x_range[1], 200)
    ys = -(constant_term + coef_x * xs) / coef_y
    mask = (ys >= y_range[0]) & (ys <= y_range[1])
    if mask.any():
        line_x_vals = xs[mask]
        line_y_vals = ys[mask]
        if not mask.all():
            line_note = "경계선의 일부는 이 그림 범위 밖에 있어요."
    else:
        line_note = "로지스틱 회귀의 경계선이 이 그림 범위 밖에 있어요."
elif abs(coef_x) > 1e-9:
    x_val = -constant_term / coef_x
    if x_range[0] <= x_val <= x_range[1]:
        line_x_vals = [x_val, x_val]
        line_y_vals = [y_range[0], y_range[1]]
    else:
        line_note = "로지스틱 회귀의 경계선이 이 그림 범위 밖에 있어요."
else:
    line_note = "이 속성 조합으로는 경계선을 그릴 수 없어요."

fig = go.Figure()

# 의사결정트리가 나눈 영역(옅은 색)
fig.add_trace(
    go.Heatmap(
        x=xx,
        y=yy,
        z=Z_dt,
        showscale=False,
        zmin=0,
        zmax=1,
        colorscale=[[0, "#cfe3ff"], [1, "#ffd0d0"]],
        opacity=0.35,
        hoverinfo="skip",
        name="의사결정트리 영역",
    )
)

# 테스트 데이터 점
for label_val, label_name, color in [(0, "뇌졸중 없음", "#1f77b4"), (1, "뇌졸중 있음", "#d62728")]:
    subset = test_df[test_df["stroke"] == label_val]
    fig.add_trace(
        go.Scatter(
            x=subset[x_feature],
            y=subset[y_feature],
            mode="markers",
            name=label_name,
            marker=dict(size=6, color=color, opacity=0.75),
        )
    )

# 로지스틱 회귀 경계선
if len(line_x_vals) > 0:
    fig.add_trace(
        go.Scatter(
            x=line_x_vals,
            y=line_y_vals,
            mode="lines",
            name="로지스틱 회귀 경계선(0.5)",
            line=dict(color="black", width=2, dash="dash"),
        )
    )

fig.update_layout(
    title="테스트 데이터와 분류 경계",
    xaxis_title=FEATURE_LABELS[x_feature],
    yaxis_title=FEATURE_LABELS[y_feature],
    xaxis=dict(range=x_range),
    yaxis=dict(range=y_range),
)

st.plotly_chart(fig, use_container_width=True)

if line_note:
    st.write(f"ℹ️ {line_note}")

st.divider()

# ---------- 6. 의사결정트리 가지 그림 ----------
st.markdown("### 4️⃣ 의사결정트리가 던진 질문")

tree = model_dt.tree_
feature_labels_ordered = [FEATURE_LABELS[f] for f in selected_features]


def build_tree_dot(tree, feature_labels_ordered):
    lines = [
        "digraph Tree {",
        'node [shape=box, style="filled", fontname="Malgun Gothic", fontsize=11];',
        'edge [fontname="Malgun Gothic", fontsize=10];',
    ]
    for i in range(tree.node_count):
        samples = int(tree.n_node_samples[i])
        value = tree.value[i][0]
        count0 = int(value[0])
        count1 = int(value[1])
        ratio1 = (count1 / samples * 100) if samples > 0 else 0.0
        is_leaf = tree.children_left[i] == tree.children_right[i]

        if is_leaf:
            predicted = 1 if count1 > count0 else 0
            answer_text = "뇌졸중" if predicted == 1 else "아님"
            label = (
                f"훈련용 {samples}명\\n"
                f"뇌졸중 {count1}명 ({ratio1:.1f}%)\\n"
                f"답: {answer_text}"
            )
            color = "#ffd0d0" if predicted == 1 else "#cfe3ff"
        else:
            feat_name = feature_labels_ordered[tree.feature[i]]
            threshold = tree.threshold[i]
            label = (
                f"{feat_name} <= {threshold:.2f} ?\\n"
                f"훈련용 {samples}명\\n"
                f"뇌졸중 {count1}명 ({ratio1:.1f}%)"
            )
            color = "#f2f2f2"

        lines.append(f'{i} [label="{label}", fillcolor="{color}"];')

    for i in range(tree.node_count):
        left = tree.children_left[i]
        right = tree.children_right[i]
        if left != -1:
            lines.append(f'{i} -> {left} [label="예"];')
        if right != -1:
            lines.append(f'{i} -> {right} [label="아니요"];')

    lines.append("}")
    return "\n".join(lines)


dot_str = build_tree_dot(tree, feature_labels_ordered)
st.graphviz_chart(dot_str)

leaf_indices = [i for i in range(tree.node_count) if tree.children_left[i] == -1]
n_leaf = len(leaf_indices)
n_leaf_negative = sum(
    1 for i in leaf_indices if tree.value[i][0][0] >= tree.value[i][0][1]
)

used_feature_indices = sorted(
    {tree.feature[i] for i in range(tree.node_count) if tree.feature[i] != -2}
)
used_features = [selected_features[idx] for idx in used_feature_indices]

st.write(f"- 답을 내는 마디(잎)는 모두 {n_leaf}칸이고, 그중 {n_leaf_negative}칸은 '아님'이라고 답해요.")

if used_features:
    for f in used_features:
        st.write(f"- 이 나무는 '{FEATURE_LABELS[f]}'을(를) 실제로 물어봤어요.")
else:
    st.write("- 이 나무는 아무 속성도 묻지 않고 바로 답을 냈어요.")
