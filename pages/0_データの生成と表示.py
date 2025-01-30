import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Streamlitアプリの設定
st.title("y = ax + b + e に従うデータ生成")

# サイドバーでパラメータを入力
st.sidebar.header("パラメータ設定")
step_num = st.sidebar.number_input("ステップ数", min_value=0.001, value=0.1)
a = st.sidebar.number_input("a (傾き)", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
b = st.sidebar.number_input("b (切片)", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
n = st.sidebar.number_input("データ点の数", min_value=10, max_value=100, value=100, step=10)
x_min = st.sidebar.number_input("xの最小値", min_value=-100.0, max_value=100.0, value=0.0, step=1.0)
x_max = st.sidebar.number_input("xの最大値", min_value=-100.0, max_value=100.0, value=10.0, step=1.0)

# データを生成
if st.button("データを生成"):
    x = np.linspace(x_min, x_max, n)
    e = np.random.normal(0, 3, n)
    y = a * x + b + e

    sample_data_df = pd.DataFrame({
        "x": x,
        "y": y,
        "a": a,  # 定数を列に入れる
        "b": b  # 定数を列に入れる
    })
else :
    st.warning("データを生成してください．")

if 'sample_data_df' in globals():
    # プロットを作成
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Generated data", alpha=0.7)
    ax.plot(x, a * x + b, color="red", label="y = ax + b")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Generated Data Following y = ax + b + e")
    # Streamlitに表示
    st.pyplot(fig)

    display_col01 = st.columns([3,1])
    with display_col01[0]:
        st.write("記録されたデータの散布図")
        x_input = pd.array(sample_data_df["x"])
        y_input = pd.array(sample_data_df["y"])
        # Figure 作成
        fig = go.Figure()
        # 散布図
        fig.add_trace(go.Scatter(x=x_input, y=y_input, mode='markers', marker=dict(color='yellow', size=8), name="散布図"))
        # Streamlit で表示
        st.plotly_chart(fig)

    with display_col01[1]:
        # データを表示
        st.write("記録されたデータ")
        st.dataframe(sample_data_df)
        st.session_state.data_df = sample_data_df
        # CSVとしてダウンロード
        import io
        csv = io.StringIO()
        np.savetxt(csv, sample_data_df, delimiter=",", header="x,y", comments="")
        st.download_button(
            label="データをCSVとしてダウンロード",
            data=csv.getvalue(),
            file_name="generated_data.csv",
            mime="text/csv"
        )