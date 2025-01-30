import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.header("最急降下法を用いたパラメータの最適化",divider="rainbow")
"""  """

input_data_df = st.session_state.mopt_datas
# データを取得
x = input_data_df["x"].to_numpy()
y = input_data_df["y"].to_numpy()
# RSS 関数
def f_RSS(x, y, p, q):
    y_pred = p * x + q
    rss = np.sum((y - y_pred) ** 2)
    return rss


st.subheader("パラメータの最適化の実行",divider="orange")

# 最大反復回数
num_iter_max = 10000
alpha = 0.00001  # 学習率
# 結果を保存する配列
popt_history_array = np.zeros((num_iter_max, 3))
# 初期値をセット
popt_history_array[0, 0] = input_data_df["opted_p"].to_numpy()[0]
popt_history_array[0, 1] = input_data_df["opted_q"].to_numpy()[0]

# 解析的に勾配を計算する関数
def compute_gradients(x, y, p, q):
    y_pred = p * x + q
    dRSS_dp = -2 * np.sum(x * (y - y_pred))  # ∂RSS/∂p
    dRSS_dq = -2 * np.sum(y - y_pred)        # ∂RSS/∂q
    return dRSS_dp, dRSS_dq

if st.button("最適化開始"):
    # 最適化ループ
    niter = 0
    d_rss = 1
    # for niter in range(num_iter_max - 1):
    while   (niter < num_iter_max-1) and (d_rss > 1.0e-15):
        p = popt_history_array[niter, 0]
        q = popt_history_array[niter, 1]

        # 勾配を計算
        dRSS_dp, dRSS_dq = compute_gradients(x, y, p, q)

        # 勾配降下による更新
        p_new = p - alpha * dRSS_dp
        q_new = q - alpha * dRSS_dq

        # 更新後の値を保存
        popt_history_array[niter + 1, 0] = p_new
        popt_history_array[niter + 1, 1] = q_new
        popt_history_array[niter + 1, 2] = f_RSS(x, y, p_new, q_new)
        
        niter +=1
        d_rss = abs(f_RSS(x, y, p_new, q_new) - f_RSS(x, y, p, q))

    popt_history_array = popt_history_array[1:niter]
    # DataFrame に変換
    popt_history_df = pd.DataFrame(popt_history_array, columns=["p_hist", "q_hist", "RSS_hist"])

    disp_col01 = st.columns([2,1])

    with disp_col01[0]:
        st.write(f"最適化回数とRSSの変化")
        # Figure 作成
        fig = go.Figure()
        # 散布図
        fig.add_trace(go.Scatter(x=popt_history_df.index, y=popt_history_df["RSS_hist"]
                                    , mode='markers', marker=dict(color='yellow', size=4)
                                    , name="RSSの変化"
                                    ))
        st.plotly_chart(fig)
            
    with disp_col01[1]:
        st.write(f"データの確認")
        st.dataframe(popt_history_df)

"""   """
st.subheader("最適化の結果",divider="orange")

tab_list = ["最終的な結果の表示"
            ,"途中のアニメーション"]
selected_tab1, selected_tab2 = st.tabs(tab_list)
with selected_tab1:
    try:
        # Figure 作成
        fig_opt = go.Figure()
        # 散布図
        fig_opt.add_trace(go.Scatter(x=x, y=y
                                    , mode='markers', marker=dict(color='yellow', size=4)
                                    , name="散布図"
                                    ))
        # 直線
        # DataFrame の最後の行を取得
        last_row = popt_history_df.iloc[-1]

        # 1列目と2列目のデータを NumPy の float として取得
        p, q = last_row.iloc[0], last_row.iloc[1]

        # NumPy の float 型に変換
        p_opt, q_opt = np.float64(p), np.float64(q)
        y_pred = p_opt*x + q_opt
        fig_opt.add_trace(go.Scatter(x=x, y=y_pred, mode='lines', line=dict(color='magenta', width=2), name="直線"))
        # グラフのタイトルを設定
        fig_opt.update_layout(title="最適化後のデータの散布図とy=px+qのグラフ")
        # Streamlit で表示
        st.plotly_chart(fig_opt)
    except:
        st.warning("まだ最適化が終わっていません．")
with selected_tab2:
    try:

        # アニメーションのフレームリスト
        frames = []

        # 最初の散布図の作成
        fig_opt = go.Figure()

        # 散布図の追加（最初のデータ）
        fig_opt.add_trace(go.Scatter(x=x, y=y, mode='markers', name='データ', marker=dict(color="blue")))

        # 初期の直線を追加
        initial_p = np.float64(popt_history_df.iloc[0]["p_hist"])
        initial_q = np.float64(popt_history_df.iloc[0]["q_hist"])
        initial_y_pred = initial_p * x + initial_q

        # 初期直線を描く
        line_trace = go.Scatter(x=x, y=initial_y_pred, mode="lines", name="回帰直線", line=dict(color="red"))
        fig_opt.add_trace(line_trace)

        # フレームを作成して直線の更新を行う
        for i in range(1, len(popt_history_df)):
            p = np.float64(popt_history_df.iloc[i]["p_hist"])
            q = np.float64(popt_history_df.iloc[i]["q_hist"])
            y_pred = p * x + q  # 直線の計算

            # フレームに新しい直線データを追加
            frame = go.Frame(
                data=[go.Scatter(x=x, y=y, mode="markers", name="データ", marker=dict(color="blue")),
                    go.Scatter(x=x, y=y_pred, mode="lines", name="回帰直線", line=dict(color="red"))],
                name=f"Step {i+1}"
            )
            frames.append(frame)

        # アニメーションの設定
        fig_opt.update_layout(
            title="回帰直線の変化 (アニメーション)",
            xaxis=dict(title="x"),
            yaxis=dict(title="y"),
            updatemenus=[{
                "type": "buttons",
                "direction": "left",
                "x": 0.5,
                "y": -0.2,
                "xanchor": "center",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "再生",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 2.5, "redraw": True}, "fromcurrent": True}]
                    },
                    {
                        "label": "停止",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                    }
                ]
            }],
            sliders=[{
                "currentvalue": {
                    "prefix": "Frame: ",
                    "visible": True,
                    "xanchor": "center",
                },
                "steps": [
                    {"args": [
                        [f"Step {i+1}"],
                        {"frame": {"duration": 2.5, "redraw": True}, "mode": "immediate"}
                    ], "label": f"Step {i+1}", "method": "animate"}
                    for i in range(len(popt_history_df))
                ]
            }]
        )

        # アニメーション用のフレームを設定
        fig_opt.frames = frames

        st.plotly_chart(fig_opt)
    except : 
        st.warning("まだ最適化が終わっていません．")