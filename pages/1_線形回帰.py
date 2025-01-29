import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.header("線形回帰式の推定", divider="rainbow")
input_data_df = st.session_state.data_df


st.subheader("データの確認",divider="orange")
st.dataframe(input_data_df[["x","y"]])


st.subheader("データ分析",divider="orange")


# Streamlit UI
st.title("回帰モデル y = px + q の 残差平方和（RSS）最適化")
st.write("スライダーで p, q を調整し、モデルの誤差を確認しよう！")

# p, q のスライダー設定
p = st.slider("傾き p", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
q = st.slider("切片 q", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)


x = pd.array(input_data_df["x"])
y = pd.array(input_data_df["y"])

# モデルによる予測値
y_pred = p * x + q



# 残差平方和（RSS）の計算
rss = np.sum((y - y_pred) ** 2)

# 残差平方和を表示
st.write(f"現在の RSS（残差平方和）: **{rss:.2f}**")

# セッション状態に試行データを記録
if "history" not in st.session_state:
    st.session_state.history = []  # 初期化

# 試行回数を最大100回に制限
if st.button("記録する"):
    if len(st.session_state.history) >= 100:
        st.session_state.history.pop(0)  # 最も古いデータを削除
    st.session_state.history.append({"p": p, "q": q, "RSS": rss})

# プロット1: 3Dプロット（p, q, RSS）
if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)

    fig = px.scatter_3d(
        df_history,
        x="p",
        y="q",
        z="RSS",
        color="RSS",
        color_continuous_scale="RdBu",  # 修正：plotly対応のカラースケールに変更
        title="試行ごとの (p, q, RSS) 3D プロット"
    )
    
    fig.update_traces(marker=dict(size=6))  # 点のサイズ調整
    fig.update_layout(
        scene=dict(
            xaxis_title="p (傾き)",
            yaxis_title="q (切片)",
            zaxis_title="RSS"
        )
    )

    st.plotly_chart(fig)

    # 履歴データの表示
    st.subheader("試行履歴（最新 10 件）")
    st.write(df_history.tail(10))

     # CSV ダウンロード
    csv = df_history.to_csv(index=False).encode("utf-8")
    st.download_button("CSV をダウンロード", data=csv, file_name="regression_history.csv", mime="text/csv")