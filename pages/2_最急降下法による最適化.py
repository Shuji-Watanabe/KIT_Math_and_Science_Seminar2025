import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.header("最急降下法を用いたパラメータの最適化",divider="rainbow")
"""  """

input_data_df = st.session_state.mopt_datas
org_data_df = st.session_state.data_df
# データを取得
x = org_data_df["x"].to_numpy()
y = org_data_df["y"].to_numpy()
# RSS 関数
def f_RSS(x, y, p, q):
    y_pred = p * x + q
    rss = np.sum((y - y_pred) ** 2)
    return rss


st.subheader("パラメータの最適化の実行",divider="orange")

# 最大反復回数
num_iter_max = 50000
alpha = 1.0e-5  # 学習率
# 結果を保存する配列
popt_history_array = np.zeros((num_iter_max, 4))
# 初期値をセット
popt_history_array[0, 0] = input_data_df["opted_p"].to_numpy()[0]
popt_history_array[0, 1] = input_data_df["opted_q"].to_numpy()[0]
popt_history_array[0, 2] = input_data_df["opted_RSS"].to_numpy()[0]
# 解析的に勾配を計算する関数
def compute_gradients(x, y, p, q):
    y_pred = np.float64(p) * x + np.float64(q) 
    dRSS_dp = -np.float64(2) * np.sum(x * (y - y_pred))  # ∂RSS/∂p
    dRSS_dq = -np.float64(2) * np.sum(y - y_pred)        # ∂RSS/∂q
    return dRSS_dp, dRSS_dq

if "optimization start" not in st.session_state:
    st.session_state["optimization start"] = False

if st.button("最適化開始"):
    st.session_state["optimization start"] = True

if st.session_state["optimization start"]:
    # 最適化ループ
    niter = 0
    d_rss = 1
    # for niter in range(num_iter_max - 1):
    while   (niter < num_iter_max-1) and (d_rss > 1.0e-10) :
        p = popt_history_array[niter, 0]
        q = popt_history_array[niter, 1]

        # 勾配を計算
        dRSS_dp, dRSS_dq = compute_gradients(x, y, p, q)

        # 勾配降下による更新
        p_new = p - alpha * dRSS_dp
        q_new = q - alpha * dRSS_dq
        d_rss = abs(f_RSS(x, y, p_new, q_new) - f_RSS(x, y, p, q))

        # 更新後の値を保存
        popt_history_array[niter + 1, 0] = p_new
        popt_history_array[niter + 1, 1] = q_new
        popt_history_array[niter + 1, 2] = f_RSS(x, y, p_new, q_new)
        popt_history_array[niter + 1, 3] = d_rss
        
        niter +=1
        
    popt_history_array = popt_history_array[0:niter]
    

    ini_data = popt_history_array[0]
    ini_RSS=f_RSS(x, y, np.float64(ini_data[0]), np.float64(ini_data[1]))
    ini_dRSS_dp, ini_dRSS_dq = compute_gradients(x, y, ini_data[0], ini_data[1])
    ini_p_new = ini_data[0]- alpha * ini_dRSS_dp
    ini_q_new = ini_data[1]- alpha * ini_dRSS_dq
    ini_d_rss = abs(f_RSS(x, y, ini_p_new, ini_q_new ) - f_RSS(x, y, ini_data[0], ini_data[1]))
    
    
    # DataFrame に変換
    popt_history_df = pd.DataFrame(popt_history_array, columns=["p_hist", "q_hist", "RSS_hist","d_ress_hist"])

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



    tmp_y_pred = np.float64(ini_data[0]) * np.float64(x) + np.float64(ini_data[1])
    st.write("最適化前")
    disp_res_col = st.columns([1]*5)
    with disp_res_col[0]:
        st.metric("最適化回数",f"{0}")
    with disp_res_col[1]:
        st.metric("$\|d(rss)\|$",f"{ini_d_rss:2.1e}")
    with disp_res_col[2]:
        st.metric("RSS",f"{ini_RSS: .3f}")
    with disp_res_col[3]:
        st.metric("$~p_{{\\rm opt}}~$",f"{ini_data[0]: .3f}")
    with disp_res_col[4]:
        st.metric("$~q_{{\\rm opt}}~$",f"{ini_data[1]: .3f}")

    st.write("最適化後")
    disp_res_col = st.columns([1]*5)
    with disp_res_col[0]:
        st.metric("最適化回数",f"{niter}")
    with disp_res_col[1]:
        st.metric("$\|d(rss)\|$",f"{d_rss:2.1e}")
    with disp_res_col[2]:
        st.metric("RSS",f"{f_RSS(x, y, p_new, q_new): .3f}")
    with disp_res_col[3]:
        st.metric("$~p_{{\\rm opt}}~$",f"{p_new: .3f}")
    with disp_res_col[4]:
        st.metric("$~q_{{\\rm opt}}~$",f"{q_new: .3f}")
    


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
    # import numpy as np
    # import plotly.graph_objects as go
    # import streamlit as st

    # セッションステートの初期化
    if "animation_running" not in st.session_state:
        st.session_state["animation_running"] = False

    if st.button("アニメーションの作成"):
        # 事前にプロット用データを生成
        frame_data = []
        popt_history_df = popt_history_df.head(500)
        for i in range(len(popt_history_df)):
            y_pred = popt_history_df.iloc[i]["p_hist"] * x + popt_history_df.iloc[i]["q_hist"]
            frame_data.append(y_pred)

        # 初期プロット
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='データ', marker=dict(color='blue')))
        fig.add_trace(go.Scatter(x=x, y=frame_data[0], mode='lines', name='回帰直線', line=dict(color='red')))

        # フレームの追加
        frames = [
            go.Frame(
                data=[
                    go.Scatter(x=x, y=y, mode='markers', name='データ', marker=dict(color='blue')),
                    go.Scatter(x=x, y=frame_data[i], mode='lines', name='回帰直線', line=dict(color='red'))
                ],
                name=f"Step {i+1}"
            )
            for i in range(len(frame_data))
        ]
        fig.frames = frames

        # レイアウトとアニメーション設定
        fig.update_layout(
            title="回帰直線の変化 (アニメーション)",
            xaxis=dict(title="x"),
            yaxis=dict(title="y"),
            updatemenus=[{
                "type": "buttons",
                "direction": "left",
                "x": 1.05,
                "y": 0,
                "xanchor": "right",
                "yanchor": "bottom",
                "buttons": [
                    {
                        "label": "再生",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 60, "redraw": True}, "fromcurrent": True}]
                    },
                    {
                        "label": "停止",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                    }
                ]
            }],
            sliders=[{
                "currentvalue": {"prefix": "Frame: ", "visible": True, "xanchor": "center"},
                "steps": [
                    {"args": [[f"Step {i+1}"], {"frame": {"duration": 60, "redraw": True}, "mode": "immediate"}],
                    "label": f"Step {i+1}", "method": "animate"}
                    for i in range(len(frame_data))
                ]
            }]
        )

        # Streamlitで表示
        st.plotly_chart(fig)

#### Matplotlibバージョン
    # import matplotlib.pyplot as plt
    # import japanize_matplotlib
    # from matplotlib.animation import FuncAnimation, PillowWriter

    # # セッションステートの初期化
    # if "animation_running" not in st.session_state:
    #     st.session_state["animation_running"] = False

    # # 事前にプロット用データを生成
    # frame_data = []
    # popt_history_df = popt_history_df.head(500)
    # for i in range(len(popt_history_df)):
    #     y_pred = popt_history_df.iloc[i]["p_hist"] * x + popt_history_df.iloc[i]["q_hist"]
    #     frame_data.append(y_pred)

    # # プロットの準備
    # fig, ax = plt.subplots()
    # sc = ax.scatter(x, y, color='blue', label='データ')
    # line, = ax.plot(x, frame_data[0], color='red', label='回帰直線')
    # frame_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    # ax.set_title("回帰直線の変化 (アニメーション)")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.legend()

    # # アニメーション関数
    # def update(frame):
    #     line.set_ydata(frame_data[frame])
    #     frame_text.set_text(f"Frame: {frame+1}/{len(frame_data)}")
    #     return line, frame_text

    # # ボタンで再生・停止
    # col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("再生"):
    #         st.session_state["animation_running"] = True

    # with col2:
    #     if st.button("停止"):
    #         st.session_state["animation_running"] = False

    # # アニメーションの実行制御
    # if st.session_state["animation_running"]:
    #     with st.spinner("matplotlibによるアニメ作成中"):
    #         ani = FuncAnimation(fig, update, frames=len(frame_data), interval=60, blit=True)
    #     with st.spinner("GIFアニメの作成中"):
    #         animation_path = "animation.gif"
    #         ani.save(animation_path, writer=PillowWriter(fps=30))
    #     st.image(animation_path, caption="回帰直線の変化", use_container_width=True)
    # else:
    #     st.write("アニメーションは停止中です。")


    # except : 
    #     st.warning("まだ最適化が終わっていません．")