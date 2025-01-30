import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.header("最急降下法を用いたパラメータの最適化",divider="rainbow")


input_data_df = st.session_state.mopt_datas

# データを取得
x = input_data_df["x"].to_numpy()
y = input_data_df["y"].to_numpy()

# 最大反復回数
num_iter_max = 1000
alpha = 0.01  # 学習率

# 結果を保存する配列
popt_history_array = np.zeros((num_iter_max, 3))

# 初期値をセット
popt_history_array[0, 0] = input_data_df["opted_p"].to_numpy()[0]
popt_history_array[0, 1] = input_data_df["opted_q"].to_numpy()[0]

# RSS 関数
def f_RSS(x, y, p, q):
    y_pred = p * x + q
    rss = np.sum((y - y_pred) ** 2)
    return rss

# 解析的に勾配を計算する関数
def compute_gradients(x, y, p, q):
    y_pred = p * x + q
    dRSS_dp = -2 * np.sum(x * (y - y_pred))  # ∂RSS/∂p
    dRSS_dq = -2 * np.sum(y - y_pred)        # ∂RSS/∂q
    return dRSS_dp, dRSS_dq

# 最適化ループ
for niter in range(num_iter_max - 1):
    p = popt_history_array[niter, 0]
    q = popt_history_array[niter, 1]

    # 勾配を計算
    dRSS_dp, dRSS_dq = compute_gradients(x, y, p, q)

    # 勾配降下による更新
    p_new = p - 0.01/((niter+1)) * dRSS_dp
    q_new = q - 0.01/((niter+1)) * dRSS_dq

    # 更新後の値を保存
    popt_history_array[niter + 1, 0] = p_new
    popt_history_array[niter + 1, 1] = q_new
    popt_history_array[niter + 1, 2] = f_RSS(x, y, p_new, q_new)

# DataFrame に変換
popt_history_df = pd.DataFrame(popt_history_array, columns=["p_hist", "q_hist", "RSS_hist"])


# Figure 作成
fig = go.Figure()
# 散布図
fig.add_trace(go.Scatter(x=popt_history_df.index, y=popt_history_df["RSS_hist"]
                            , mode='markers', marker=dict(color='yellow', size=4)
                            , name="RSSの変化"
                            ))
# 直線
fig.add_trace(go.Scatter(x=popt_history_df.index, y=popt_history_df["RSS_hist"]
                         , mode='lines', line=dict(color='magenta', width=2), name="直線"))
# グラフのタイトルを設定
fig.update_layout(title="最適化の様子")
# Streamlit で表示
st.plotly_chart(fig)
    

st.dataframe(popt_history_df)