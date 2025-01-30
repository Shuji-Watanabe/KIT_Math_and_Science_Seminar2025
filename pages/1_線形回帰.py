import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.header("線形回帰式の推定", divider="rainbow")
input_data_df = st.session_state.data_df
x_input = pd.array(input_data_df["x"])
y_input = pd.array(input_data_df["y"])


st.subheader("データ分析",divider="orange")


# セッション状態に試行データを記録


display_col02 = st.columns([3,1])
with display_col02[1]:
    if st.button("探索履歴の初期化") or ("history" not in st.session_state):
        st.session_state.history = []  # 初期化 
    # p, q のスライダー設定
    num_step = st.number_input("探索ステップ", value=0.001)
    p = st.number_input("傾き p", min_value=-5.000, max_value=5.000, value=1.000, step=num_step)
    q = st.number_input("切片 q", min_value=-5.000, max_value=5.000, value=0.000, step=num_step)
    # モデルによる予測値（Predicted value）
    y_pred = p * x_input + q
    # 残差平方和（RSS）の計算
    rss = np.sum((y_input - y_pred) ** 2)
    df_history = pd.DataFrame(st.session_state.history)
""" """
    

# 試行回数を最大1000回に制限
if len(st.session_state.history) >= 1000:
    st.session_state.history.pop(0)  # 最も古いデータを削除
st.session_state.history.append({"p": p, "q": q, "RSS": rss})

with display_col02[0]:
    # Figure 作成
    fig = go.Figure()
    # 散布図
    fig.add_trace(go.Scatter(x=x_input, y=y_input
                             , mode='markers', marker=dict(color='yellow', size=4)
                             , name="散布図"
                             ))
    # 直線
    fig.add_trace(go.Scatter(x=x_input, y=y_pred, mode='lines', line=dict(color='magenta', width=2), name="直線"))
    # グラフのタイトルを設定
    fig.update_layout(title="データの散布図とy=px+qのグラフ")
    # Streamlit で表示
    st.plotly_chart(fig)

if len(df_history) != 0:
    # RSS が最小のインデックスを取得
    min_index = int(df_history["RSS"].idxmin())
    # RSS が最小の a, b を取得
    optimal_row = df_history.loc[min_index]
    st.markdown(f"今のRSSの値と探索範囲におけるRSSの最小値，最小値における傾き$~p_{{\\rm m.opt}}~$と$~y~$切片$~q_{{\\rm m.opt}}~$")
    disp_res_col = st.columns([1]*4)
    with disp_res_col[0]:
        st.metric("今のRSS",f"{rss:.3f}")
    with disp_res_col[1]:
        st.metric("RSS最小値",f"{optimal_row[2]: .3f}")
    with disp_res_col[2]:
        st.metric("$~p_{{\\rm m.opt}}~$",f"{optimal_row[0]: .3f}")
    with disp_res_col[3]:
        st.metric("$~q_{{\\rm m.opt}}~$",f"{optimal_row[1]: .3f}")


display_col03 = st.columns([1,1])
with display_col03[0]:
    # SessionState からデータ取得
    

    # 3D 散布図の作成
    fig = px.scatter_3d(df_history,
                        x="p",y="q",z="RSS"
                        ,color="RSS"
                        ,color_continuous_scale="Rainbow_r"
                        ,title="試行ごとの (p, q, RSS) 3D プロット"
    )

    fig.update_traces(marker=dict(size=2))  # 点のサイズ調整
    fig.update_layout(
        scene=dict(xaxis_title="p (傾き)",
                   yaxis_title="q (切片)",
                   zaxis_title="RSS"
        )
    )

    # Streamlit で表示
    st.plotly_chart(fig)
with display_col03[1]:
    p_vals = np.linspace(optimal_row[0]- 0.05,  optimal_row[0] + 0.05, 50)
    q_vals = np.linspace(optimal_row[1] - 0.01, optimal_row[1] + 0.01, 50)
    Mesh_p, Mesh_q = np.meshgrid(p_vals, q_vals)
    # 各 (p, q) に対する RSS を計算
    RSS = np.zeros((len(q_vals), len(p_vals))) # RSS の格納用配列
    for i in range(p_vals.shape[0]):  # q のループ
        for j in range(q_vals.shape[0]):  # p のループ
            y_pred = p_vals[i] * x_input + q_vals[j]  # y = px + q の予測値
            RSS[i, j] = np.sum((y_input - y_pred) ** 2)  # RSS 計算
    # Plotly で 3D プロット
    fig2 = go.Figure()
    # 3D サーフェスプロットを追加
    fig2.add_trace(go.Surface(
        x=p_vals, y=q_vals, z=RSS, 
        colorscale="Rainbow_r", 
        opacity=0.6,
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
    ))

    # レイアウト設定
    fig2.update_layout(
        title="最小値周辺のRSS の 3D プロット",
        scene=dict(
            xaxis_title="p (傾き)",
            yaxis_title="q (切片)",
            zaxis_title="RSS"
        )
    )
    # **(a, b) の点を追加**
    fig2.add_trace(go.Scatter3d(
        x=[optimal_row[0]], y=[optimal_row[1]], z=[optimal_row[2]],  # 単一の点
        mode='markers',
        marker=dict(color='red', size=6, symbol='circle'),
        name=f"Point ($p_{{\\rm m.opt}}$, q_{{\\rm m.opt}})"
    ))
    # Streamlit で表示
    st.plotly_chart(fig2)

tmp_data_df = pd.DataFrame({ "opted_p":[optimal_row[0]]
                            ,"opted_q":[optimal_row[1]]
                            ,"opted_RSS":[optimal_row[2]]
                            })
manual_opt_results = pd.concat([input_data_df,df_history,tmp_data_df],axis=1)
st.dataframe(manual_opt_results)
st.session_state.mopt_datas = manual_opt_results

