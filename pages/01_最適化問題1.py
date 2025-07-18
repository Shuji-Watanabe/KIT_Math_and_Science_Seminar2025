import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- Streamlit UI ---
st.title("最適化問題１")
# --- パラメータ定義 ---
st.header("初期パラメータの設定",divider="violet")
with st.expander("設定の確認・変更") :
    input_cols = st.columns([1]*4)
    with input_cols[0] :
        x_ini = st.number_input("通過点 $A$ の $x$ 座標 $\\rm [m]$"
                                , min_value=0, max_value=0
                                ,value=0)
        x_ini = float(x_ini)
        y_ini = st.number_input("通過点 $A$ の $y$ 座標 $\\rm [m]$"
                                ,min_value=float(0)
                                ,value=1.0)
        y_start = float(y_ini)  # 始点の y 座標 (>0)
        

    with input_cols[1] :
        xt    = st.number_input("通過点 $B$ の $x$ 座標 $\\rm [m]$"
                                , min_value=float(0)
                                ,value=3.0)
        xt = float(xt)
        yt    = st.number_input("通過点 $B$ の $y$ 座標 $\\rm [m]$"
                                ,max_value=float(0)
                                ,value=-1.0)
        yt = float(yt)          # 終点座標 (yt<0)
        
    with input_cols[2] :
        v1 = st.number_input("領域1での速さ $\\rm [m/s]$"
                                ,min_value=float(0)
                                ,value=1.0)
        v1 = float(v1)         # 領域1 (y>=0) の速さ
    with input_cols[3] :
        v2 = st.number_input("領域2での速さ $\\rm [m/s]$"
                                ,min_value=float(0)
                                ,value=0.5)
        v2 = float(v2)         # 領域2 (y<0) の速さ
            

# --- 時間関数と勾配 ---
def time_T(X):
    """交点 X を経由したときの総移動時間 T(X)"""
    d1 = np.hypot(X, y_start) / v1
    d2 = np.hypot(xt - X, yt) / v2
    return d1 + d2

def grad_T(X):
    """T(X) の勾配 dT/dX"""
    dist1 = np.hypot(X, y_start)
    dist2 = np.hypot(xt - X, yt)
    g1 = X / (v1 * dist1)
    g2 = -(xt - X) / (v2 * dist2)
    return g1 + g2

def steepest_descent(initial_X=1.0, alpha=0.1, beta = 0.01, k=0,tol=1e-6, max_iter=1000):
    """最急降下法で最適交点を探索"""
    X = initial_X
    for _ in range(max_iter):
        g = grad_T(X)
        if abs(g) < tol:
            break
        # X -= alpha * g
        X -= alpha/(1+k*beta) * g
    return X

# --- セッションステート初期化 ---
if 'history_X' not in st.session_state:
    st.session_state.history_X = []
    st.session_state.history_T = []

def append_history():
    X_val = st.session_state.number_input_X
    T_val = time_T(X_val)
    st.session_state.history_X.append(X_val)
    st.session_state.history_T.append(T_val)
    
def optimize_with_history(initial_X, alpha, beta, tol=1e-6, max_iter=1000):
    X = initial_X
    history_X = [X]
    history_T = [time_T(X)]
    for _ in range(max_iter):
        g = grad_T(X)
        if abs(g) < tol:
            break
        # X -= alpha * g
        X -= alpha/(1+_*beta) * g
        history_X.append(X)
        history_T.append(time_T(X))
    return history_X, history_T


'___'
  
st.header("**最適化**", divider="violet")
col01 = st.columns(2)
with col01[0]:
    mode = st.radio("モード選択", ("手動最適化", "自動最適化")
                ,horizontal=True)
with col01[1]:
    # 履歴クリアボタン
    if st.button("履歴クリア"):
        st.session_state.history_X = []
        st.session_state.history_T = []



# --- 図の並列表示 ---
col1, col2 = st.columns(2)

marker_color = "red"
if mode == "手動最適化":
    with col1:
        initial_X = 1.00
        alpha = 0.01
        X= st.number_input(label="求める $x$ 座標の手動入力"
                            , min_value=0.000, max_value=5.000, step=0.001
                            , format="%.3f"
                            , on_change=append_history
                            , key="number_input_X")
        T = time_T(X)
        output_text = f"点$\\rm P$の $x$ 座標 = {X:.3f}， 移動時間 $T$ = {T:.4f}"
        # 経路プロット
        fig_path = go.Figure()
        # 領域1（濃い緑）
        fig_path.add_trace(go.Scatter(
            x=[0, X], y=[y_start, 0], mode='lines+markers',
            line=dict(color='green', width=2),
            marker=dict(color=marker_color, size=8),  # 始点
            showlegend=False
        ))
        # 領域2（青）
        fig_path.add_trace(go.Scatter(
            x=[X, xt], y=[0, yt], mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(color=marker_color, size=8),  # 終点
            showlegend=False
        ))
        # 始点・終点（xマーク、色はテーマ依存）
        fig_path.add_trace(go.Scatter(
            x=[0, xt], y=[y_start, yt], mode='markers',
            marker=dict(color=marker_color, symbol='x', size=12),
            showlegend=False
        ))
        fig_path.update_layout(
            title="移動経路", xaxis_title="x", yaxis_title="y", width=400, height=400,
            showlegend=False
        )
        st.plotly_chart(fig_path)
        st.write(output_text)

    with col2:
        code_input = st.text_input("解答表示のパスワード", "")
        # 時間履歴プロット（点のみ）
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=st.session_state.history_X,
            y=st.session_state.history_T,
            mode='markers',
            marker=dict(color='blue', size=4, symbol='cross'),
            name='履歴',
            showlegend=False
        ))
        # 最小値点（赤丸, オプション）
        if st.session_state.history_T:
            idx_min = int(np.argmin(st.session_state.history_T))
            min_T = st.session_state.history_T[idx_min]
            min_X = st.session_state.history_X[idx_min]
            fig_time.add_trace(go.Scatter(
                x=[min_X], y=[min_T],
                mode='markers',
                marker=dict(color='red', size=6, symbol='circle'),
                name='最小時間点',
                showlegend=False
            ))
            curr_X = st.session_state.history_X[-1]
            curr_T = st.session_state.history_T[-1]
            fig_time.add_trace(go.Scatter(
                                            x=[curr_X], y=[curr_T],
                                            mode='markers',
                                            marker=dict(
                                                symbol='circle-open'
                                                ,size=10
                                                ,line=dict(width=4, color='magenta')
                                            ),
                                            name='現在'
                                            , showlegend=False
                                        ))
        # 厳密解表示
        if code_input == "2525":
            X_exact = steepest_descent()  # デフォルトパラメータで計算
            fig_time.add_vline(x=X_exact, line=dict(color='orange', dash='dash'), annotation_text="数値的な解", annotation_position="top right")
        fig_time.update_layout(title="最適化の履歴", xaxis_title="x 座標", yaxis_title="移動時間 T", width=400, height=400)
        st.plotly_chart(fig_time)
        if st.session_state.history_T:
            # NumPy を使って最小値のインデックスを取得
            idx_min = int(np.argmin(st.session_state.history_T))
            
            # 最小時間とそのときの X
            min_T = st.session_state.history_T[idx_min]
            min_X = st.session_state.history_X[idx_min]
            
            st.write(f"最小移動時間 $T$ = {min_T:.4f} （$x_{{\\rm min}}$ = {min_X:.3f}）") 
        
elif mode == "自動最適化":
    from plotly.subplots import make_subplots
    # 1. 初期値と学習率の設定
    input_cols2 = st.columns([1]*3)
    with input_cols2[0]:
        initial_X = st.number_input("初期交点 X", value=float(0), format="%.3f")
    with input_cols2[1]:
        alpha     = st.number_input("学習率 α", value=0.3, format="%.2f", min_value=0.01, max_value=1.00, step=0.01)
    with input_cols2[2]:
        # 速度設定用スライダー（アニメーション速度）
        speed = st.slider("アニメーション速度 (ms/frame)", min_value=100, max_value=2000, value=500, step=100)

    # 2. スタートボタンの設定
    if st.button("Start Optimization"):
        # 3. 最急降下法で最適化（途中履歴も記録）
        hist_X, hist_T = optimize_with_history(initial_X, alpha=alpha, beta=0.01)

        # 5. アニメーション図を2列で作成（Plotly の make_subplots）
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("経路アニメーション", "時間履歴アニメーション")
        )

        # フレーム生成
        frames = []
        ymin, ymax = min(hist_T)*0.9, max(hist_T)*1.1
        for i, (X_val, T_val) in enumerate(zip(hist_X, hist_T)):
            # 左：経路
            path1 = go.Scatter(x=[0, X_val], y=[y_start, 0]
                               , mode='lines'
                               , name='領域1' 
                               , line=dict(color='blue', width=3)
                            #    , marker=dict(color=marker_color,size=10)
                               , showlegend=False)
            path2 = go.Scatter(x=[X_val, xt], y=[0, yt]
                               , mode='lines'
                               , name='領域2'
                               , line=dict(color='green', width=3)
                            #    , marker=dict(color=marker_color,size=10)
                               , showlegend=False)
            path3 = go.Scatter(x=[0, xt], y=[y_start, yt]
                               , mode='markers'
                               , marker=dict(color=marker_color, symbol='x', size=12)
                               , name='始点・終点'
                               , showlegend=False)
            # path4 = go.Scatter(x=[xt], y=[0]
            #                     , mode='markers'
            #                     , marker=dict(color=marker_color, size=12)
            #                     , name='中継点'
            #                     , showlegend=False)
            # 右：時間履歴
            time_trace = go.Scatter(
                x=hist_X[:i+1], y=hist_T[:i+1],
                mode='markers',
                marker=dict(color='blue', symbol='cross', size=5),
                name='履歴'
                , showlegend=False
            )
            # フレーム用
            frames.append(go.Frame(
                data=[path1, path2, path3, time_trace],
                name=str(i)
            ))

        # 初期フレーム設定
        fig.add_trace(frames[0].data[0], row=1, col=1)
        fig.add_trace(frames[0].data[1], row=1, col=1)
        fig.add_trace(frames[0].data[2], row=1, col=1)
        fig.add_trace(frames[0].data[3], row=1, col=2)

        # レイアウト設定
        fig.update_layout(
            width=900, height=500,
            xaxis1=dict(range=[0, xt], title="x"),
            yaxis1=dict(range=[yt, y_start], title="y"),
            xaxis2=dict(range=[0, xt], title="X"),
            yaxis2=dict(range=[ymin, ymax], title="T"),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, {"frame": {"duration": speed, "redraw": True}, "fromcurrent": True}]
                )]
            )]
        )
        fig.frames = frames

        # Step数と各ステップの X, T を下に表示
        st.markdown("### 最適化履歴")
        st.write(f"Step {i}: $x_{{{i}}}$ = {hist_X[-1]:.3f}, $T_{{{i}}}$ = {hist_T[-1]:.4f}")
        # 描画
        st.plotly_chart(fig, use_container_width=True)


