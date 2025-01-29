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


display_col02 = st.columns([1,1])
with display_col02[1]:
    # p, q のスライダー設定
    p = st.number_input("傾き p", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
    q = st.number_input("切片 q", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    # モデルによる予測値（Predicted value）
    y_pred = p * x_input + q
    # 残差平方和（RSS）の計算
    rss = np.sum((y_input - y_pred) ** 2)
with display_col02[0]:
    # 残差平方和を表示
    st.write(f"現在の RSS（残差平方和）: **{rss:.2f}**")
    if st.button("初期化") or ("history" not in st.session_state):
        st.session_state.history = []  # 初期化 
# 試行回数を最大100回に制限
# if st.button("記録する"):
if len(st.session_state.history) >= 1000:
    st.session_state.history.pop(0)  # 最も古いデータを削除
st.session_state.history.append({"p": p, "q": q, "RSS": rss})



display_col03 = st.columns([1,1])


with display_col03[0]:
    # SessionState からデータ取得
    df_history = pd.DataFrame(st.session_state.history)

    # 3D 散布図の作成
    fig = px.scatter_3d(df_history,
                        x="p",y="q",z="RSS",
                        color="RSS",
                        color_continuous_scale="Rainbow",title="試行ごとの (p, q, RSS) 3D プロット"
    )

    fig.update_traces(marker=dict(size=2))  # 点のサイズ調整
    fig.update_layout(
        scene=dict(xaxis_title="p (傾き)",
                   yaxis_title="q (切片)",
                   zaxis_title="RSS"
        )
    )

    # # 3D 曲面を作成するための p, q 範囲を決定
    # p_vals = np.linspace(df_history["p"].min(), df_history["p"].max(), 30)
    # q_vals = np.linspace(df_history["q"].min(), df_history["q"].max(), 30)
    # P, Q = np.meshgrid(p_vals, q_vals)

    # # 残差平方和（RSS）の計算
    # RSS = np.zeros_like(P)  # P と同じ形の 2D 配列を用意

    # for i in range(P.shape[0]):
    #     for j in range(P.shape[1]):
    #         p_val = P[i, j]
    #         q_val = Q[i, j]
    #         RSS[i, j] = np.sum((y_input - (p_val * x_input + q_val)) ** 2)

    # # 3D 等高線（曲面）を追加
    # fig.add_trace(
    #     go.Surface(
    #         x=P, y=Q, z=RSS,
    #         colorscale="Rainbow",
    #         opacity=0.5,  # 半透明にして散布図を見えやすくする
    #         showscale=False  # 色スケールをオフ
    #     )
    # )

    # Streamlit で表示
    st.plotly_chart(fig)
with display_col03[1]:
    # Figure 作成
    fig = go.Figure()
    # 散布図
    fig.add_trace(go.Scatter(x=x_input, y=y_input, mode='markers', marker=dict(color='yellow', size=4), name="散布図"))
    # 直線
    fig.add_trace(go.Scatter(x=x_input, y=y_pred, mode='lines', line=dict(color='magenta', width=2), name="直線"))
    # Streamlit で表示
    st.plotly_chart(fig)

display_col04 = st.columns([2,1])
with display_col04[0]:
    # RSS が最小のインデックスを取得
    min_index = int(df_history["RSS"].idxmin())

    # RSS が最小の a, b を取得
    optimal_row = df_history.loc[min_index]
    st.markdown("#### 試行結果におけるRSSの最小値")
    st.markdown(f"$${{\\rm RSS}}_{{\\rm min}} = {optimal_row[2]: .5f}$$")
    """ """
    st.markdown("#### RSSの最小値を与える線形回帰モデル")
    st.markdown(f"$$y={optimal_row[0]: .5f}x+{optimal_row[1]: .5f}$$")

with display_col04[1]:
    # 履歴データの表示
    st.write("#### 試行結果")
    st.dataframe(df_history)
    # CSV ダウンロード
    csv = df_history.to_csv(index=False).encode("utf-8")
    st.download_button("CSV をダウンロード", data=csv, file_name="regression_history.csv", mime="text/csv")


tmp_data_df = pd.DataFrame({ "opted_p":[optimal_row[0]]
                            ,"opted_q":[optimal_row[1]]
                            ,"opted_RSS":[optimal_row[2]]
                            })
manual_opt_results = pd.concat([input_data_df,df_history,tmp_data_df],axis=1)
st.dataframe(manual_opt_results)

if st.button("手動最適化の結果を保存"):
    st.session_state.mopt_datas = manual_opt_results