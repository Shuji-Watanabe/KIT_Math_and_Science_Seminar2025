import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- Streamlit UI ---
st.title("最速経路探索１")
# --- パラメータ定義 ---
y_start = 1.0      # 始点の y 座標 (>0)
xt, yt = 3.0, -1.0 # 終点座標 (yt<0)
v1 = 1.0           # 媒質1 (y>=0) の速度
v2 = 0.5           # 媒質2 (y<0) の速度

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

def steepest_descent(initial_X=1.0, alpha=0.1, tol=1e-6, max_iter=1000):
    """最急降下法で最適交点を探索"""
    X = initial_X
    for _ in range(max_iter):
        g = grad_T(X)
        if abs(g) < tol:
            break
        X -= alpha * g
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
    
def optimize_with_history(initial_X, alpha, tol=1e-6, max_iter=1000):
    X = initial_X
    history_X = [X]
    history_T = [time_T(X)]
    for _ in range(max_iter):
        g = grad_T(X)
        if abs(g) < tol:
            break
        X -= alpha * g
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
        fig_path.add_trace(go.Scatter(x=[0, X], y=[y_start, 0], mode='lines+markers', name='領域１'))
        fig_path.add_trace(go.Scatter(x=[X, xt], y=[0, yt], mode='lines+markers', name='領域２'))
        fig_path.add_trace(go.Scatter(x=[0, xt], y=[y_start, yt], mode='markers', marker=dict(symbol='x', size=10), name='始点・終点'))
        fig_path.update_layout(title="移動経路", xaxis_title="x", yaxis_title="y", width=400, height=400)
        st.plotly_chart(fig_path)
        st.write(output_text)
    with col2:
        code_input = st.text_input("$x$の最適値表示のためのパスワード", "")
        # 時間履歴プロット（点のみ）
        fig_time = go.Figure()
        # 履歴点（+マーク, 青）
        fig_time.add_trace(go.Scatter(
            x=st.session_state.history_X,
            y=st.session_state.history_T,
            mode='markers',
            marker=dict(color='blue', size=4, symbol='cross'),  # 'cross' = +
            name='履歴'
        ))

        # 最小値を赤の丸でプロット
        if st.session_state.history_T:
            idx_min = int(np.argmin(st.session_state.history_T))
            min_T = st.session_state.history_T[idx_min]
            min_X = st.session_state.history_X[idx_min]
            fig_time.add_trace(go.Scatter(
                x=[min_X], y=[min_T],
                mode='markers',
                marker=dict(color='red', size=4, symbol='circle'),
                name='最小時間点'
            ))
        # 厳密解表示
        if code_input == "20250719":
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
    initial_X = st.number_input("初期交点 X", value=1.0, format="%.3f")
    alpha     = st.number_input("学習率 α", value=0.1, format="%.4f", min_value=0.0001, max_value=1.0, step=0.0001)
    # 速度設定用スライダー（アニメーション速度）
    speed = st.slider("アニメーション速度 (ms/frame)", min_value=100, max_value=2000, value=500, step=100)

    # 2. スタートボタンの設定
    if st.button("Start Optimization"):
        # 3. 最急降下法で最適化（途中履歴も記録）
        hist_X, hist_T = optimize_with_history(initial_X, alpha)

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
            path1 = go.Scatter(x=[0, X_val], y=[y_start, 0], mode='lines+markers', name='媒質1')
            path2 = go.Scatter(x=[X_val, xt], y=[0, yt], mode='lines+markers', name='媒質2')
            path3 = go.Scatter(x=[0, xt], y=[y_start, yt], mode='markers',
                               marker=dict(symbol='x', size=5), name='始点・終点')
            # 右：時間履歴
            time_trace = go.Scatter(
                x=hist_X[:i+1], y=hist_T[:i+1],
                mode='markers',
                marker=dict(color='blue', symbol='cross', size=5),
                name='履歴'
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

        # 描画
        st.plotly_chart(fig, use_container_width=True)

        # Step数と各ステップの X, T を下に表示
        st.markdown("### 最適化履歴")
        st.write(f"Step {i}: X = {hist_X[-1]:.3f}, T = {hist_T[-1]:.4f}")


# --- 実行方法 ---
st.header("**使い方**", divider="violet")
st.markdown("""
---
- 「手動最適化」モードでスライダーを動かすと、選択した X と T の履歴が青い点でプロットされます。  
- 「履歴クリア」ボタンで履歴をリセットできます。  
- 「自動最適化」モードでは最急降下法により最適 X を探索し、結果を表示します。
""")