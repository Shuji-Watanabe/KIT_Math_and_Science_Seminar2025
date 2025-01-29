import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

# Streamlitアプリの設定
st.title("y = ax + b + e に従うデータ生成")

# サイドバーでパラメータを入力
st.sidebar.header("パラメータ設定")
a = st.sidebar.slider("a (傾き)", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
b = st.sidebar.slider("b (切片)", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
n = st.sidebar.slider("データ点の数", min_value=10, max_value=100, value=100, step=10)
x_min = st.sidebar.slider("xの最小値", min_value=-100.0, max_value=100.0, value=0.0, step=1.0)
x_max = st.sidebar.slider("xの最大値", min_value=-100.0, max_value=100.0, value=10.0, step=1.0)

# データを生成
x = np.linspace(x_min, x_max, n)
e = np.random.normal(0, 3, n)
y = a * x + b + e

sample_data_df = pd.DataFrame({
    "x": x,
    "y": y,
    "a": a,  # 定数を列に入れる
    "b": b  # 定数を列に入れる
})

st.session_state.data_df = sample_data_df

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

# データを表示
st.subheader("生成されたデータ")
data = np.column_stack((x, y))
st.write("最初の10件を表示します：")
st.write(data[:10, :])

# CSVとしてダウンロード
import io
csv = io.StringIO()
np.savetxt(csv, data, delimiter=",", header="x,y", comments="")
st.download_button(
    label="データをCSVとしてダウンロード",
    data=csv.getvalue(),
    file_name="generated_data.csv",
    mime="text/csv"
)
