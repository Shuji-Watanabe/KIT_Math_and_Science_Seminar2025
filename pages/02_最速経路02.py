import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve

# ---------------------------------
# コア関数定義
# ---------------------------------
def nonuniform_nodes(a, N, gamma_nodes=2.5):
    u = np.linspace(0, 1, N+1)
    return a * u**gamma_nodes

# 手動モード：各区間を等加速度運動とみなし移動時間を計算
def travel_time_manual(yk, xk, g=9.81):
    total_t = 0.0
    v_prev = 0.0
    for i in range(len(xk) - 1):
        dx = xk[i+1] - xk[i]
        dy = yk[i+1] - yk[i]
        s = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        a_inst = g * np.sin(theta)
        if abs(a_inst) < 1e-8:
            # 加速度ほぼゼロなら等速
            t = s / max(v_prev, 1e-6)
        else:
            disc = v_prev**2 + 2 * a_inst * s
            t = (-v_prev + np.sqrt(max(disc, 0.0))) / a_inst
        total_t += t
        v_prev += a_inst * t
    return total_t

# Simpson＋Sigmoid＋Momentum 自動最適化用
def travel_time_simpson(yk, xk, g=9.81, eps=1e-6):
    N = len(xk) - 1
    total = 0.0
    for i in range(N):
        dx, dy = xk[i+1] - xk[i], yk[i+1] - yk[i]
        ds = np.hypot(dx, dy)
        yi, yip1 = max(eps, yk[i]), max(eps, yk[i+1])
        ymid = max(eps, 0.5*(yi + yip1))
        vi = np.sqrt(2 * g * yi)
        vmid = np.sqrt(2 * g * ymid)
        vip1 = np.sqrt(2 * g * yip1)
        if i == 0:
            total += ds / vmid
        else:
            total += ds * (1/vi + 4/vmid + 1/vip1) / 6
    return total

def compute_gradient_simpson(yk, xk, eps=1e-6):
    N = len(xk) - 1
    grad = np.zeros_like(yk)
    for i in range(1, N):
        ykp, ykm = yk.copy(), yk.copy()
        ykp[i] += eps; ykm[i] -= eps
        grad[i] = (travel_time_simpson(ykp, xk) - travel_time_simpson(ykm, xk)) / (2 * eps)
    return grad

def steepest_descent_simpson_sigmoid_momentum(a, b, N,yk=None,
                                              gamma_nodes=1.0,
                                              kappa=5, delta=3,
                                              alpha=2, beta=0.0001,
                                              momentum=0.8,
                                              max_iter=6000, tol=1e-8):
    if delta is None:
        delta = 2 - 0.01 * N
    xk = nonuniform_nodes(a, N, gamma_nodes)
    if yk is None:
        yk = b * (xk / a)
    else :
        yk=yk
    v = np.zeros_like(yk)
    u = xk / a
    lr_factor = 1 / (1 + np.exp(-kappa * (u - delta)))
    lr_factor[0] = lr_factor[-1] = 0.0

    for k in range(max_iter):
        grad = compute_gradient_simpson(yk, xk)
        step = alpha / (1 + beta * k)
        update = step * lr_factor * grad
        v = momentum * v + update
        yk[1:-1] -= v[1:-1]
        yk = np.clip(yk, 0.0, None)
        yk[0], yk[-1] = 0.0, b
        if np.linalg.norm(v[1:-1]) < tol:
            break
    return xk, yk

def solve_theta(a, b):
    return fsolve(lambda th: (th - np.sin(th)) / (1 - np.cos(th)) - a/b, np.pi)[0]

def exact_cycloid(a, b, num=300):
    th = np.linspace(0, solve_theta(a, b), num)
    R = b / (1 - np.cos(th[-1]))
    return R*(th - np.sin(th)), R*(1 - np.cos(th))

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.title("Simpson+Sigmoid+Momentum による最速降下曲線")

# 固定パラメータ
a, b = 0.6, 0.3

# セッションステート初期化
if 'saved_datasets' not in st.session_state:
    st.session_state.saved_datasets = []

col2_0 = st.columns([2,1,1])

with col2_0[0]:
    # モード選択
    mode = st.radio("モード選択", ["手動最適化", "自動最適化"],horizontal=True)
with col2_0[1]:
    # 共通：分割数 N
    N = st.number_input("分割数 N", min_value=1, max_value=20, value=6, step=1)
    xk = nonuniform_nodes(a, N, gamma_nodes=1.0)
with col2_0[2]:
    # パスワード入力
    code_input = st.text_input("解答表示のパスワード")

if mode == "手動最適化":
    st.markdown("### 各節点の y 座標入力")
    # 保存済みデータ選択
    if st.session_state.saved_datasets:
        idx = st.selectbox("保存済みデータを選択", list(range(len(st.session_state.saved_datasets))),
                           format_func=lambda i: f"データ {i}", key="select_dataset")
        yk = st.session_state.saved_datasets[idx]['y']
    else:
        yk = []
    Col_num = 5
    Input_cols = st.columns( Col_num )
    
    for j in range(len(xk)):
        
        if j == 0:
            with Input_cols[0]:
                yj = st.number_input(
                    f"y[{0}]", value=0
                    ,min_value=0,max_value=0, key=f"manual_y_{0}"
                )
            yk.append(0.0)
        elif j == len(xk)-1:
            with Input_cols[ j%Col_num]:
                yj = st.number_input(
                    f"y[{j}]", value=b,
                    format="%.4f",step=0.001, min_value=b,max_value=b, key=f"manual_y_{j}"
                )
            yk.append(b)
        else:
            default = b * (xk[j]/a)
            with Input_cols[ j%Col_num]:
                yj = st.number_input(
                    f"y[{j}]", value=float(default),
                    format="%.4f", step=0.01, key=f"manual_y_{j}"
                )
            yk.append(yj)
    # 移動時間計算（等加速度モデル）
    T_manual = travel_time_manual(yk, xk)
        

    # 右カラム：y座標入力＋保存・リセット
    # 保存・リセット
    save_col, reset_col = st.columns(2)
    with save_col:
        if st.button("保存"):
            st.session_state.saved_datasets.append({'y': yk.copy(), 'T': T_manual})
            st.success(f"データを保存しました (T={T_manual:.6f} 秒)")
    with reset_col:
        if st.button("リセット"):
            for j in range(1, len(xk)-1):
                st.session_state.pop(f"manual_y_{j}", None)
            del st.session_state.saved_datasets
            # リセット後は自動で再レンダリングされます
            
    if "saved_datasets" not in st.session_state:
        st.session_state["saved_datasets"] = []
    col1, col2 = st.columns([1,1])
    # 左カラム：プロット
    if st.session_state.saved_datasets:
        with col2:
            # 保存データの移動時間プロット
            times = [d['T'] for d in st.session_state.saved_datasets]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=list(range(len(times))), y=times,
                mode='lines+markers', name='保存時間'
            ))
            fig2.update_layout(
                title="保存データの移動時間",
                xaxis_title="データ番号", yaxis_title="移動時間 T"
            )
            st.plotly_chart(fig2, use_container_width=True)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=xk, y=yk, mode='lines+markers', name='手動解'
            ))
            # パスワード正解時のみ厳密解を表示
            if code_input == "2501":
                xc, yc = exact_cycloid(a, b)
                fig.add_trace(go.Scatter(
                    x=xc, y=yc, mode='lines',
                    name='厳密解 (Cycloid)',
                    line=dict(color='orange', dash='dash')
                ))
            fig.update_layout(
                title=f"手動解 (T = {T_manual:.6f} 秒)",
                xaxis_title="x", yaxis_title="y",
                yaxis_autorange='reversed'
            )
            fig.update_layout(
            legend=dict(
                x=1,
                y=1,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.5)",  # 背景を少し透過させても見やすい
                bordercolor="black",
                borderwidth=1
            ))
            st.plotly_chart(fig, use_container_width=True)

    else :
        st.write("保存して")

elif mode == "自動最適化":
    st.markdown("### 自動最適化：最小移動時間データを初期値に使用")

    # 最小Tの保存データを初期値として
    if st.session_state.saved_datasets:
        best = min(st.session_state.saved_datasets, key=lambda d: d['T'])
        yk_init = best['y']
    else:
        yk_init = [0.0] + [b*(x/a) for x in xk[1:-1]] + [b]

    # st.write("初期解（保存データから）:", [f"{y:.4f}" for y in yk_init])

    if st.button("Start Optimization"):
        xk_opt, yk_opt = steepest_descent_simpson_sigmoid_momentum(a, b, N, yk=yk_init)
        T_num = travel_time_simpson(yk_opt, xk_opt)
        fig = go.Figure()
        # 初期手動経路
        fig.add_trace(go.Scatter(
            x=xk, y=yk_init, mode='lines+markers', name='初期解'
        ))
        # 最適化経路
        fig.add_trace(go.Scatter(
            x=xk_opt, y=yk_opt, mode='lines+markers', name='最適化経路'
        ))
        # パスワード正解時のみ厳密解を表示
        if code_input == "2501":
            xc, yc = exact_cycloid(a, b)
            fig.add_trace(go.Scatter(
                x=xc, y=yc, mode='lines',
                name='厳密解 (Cycloid)',
                line=dict(color='orange', dash='dash')
            ))
        fig.update_layout(
            title=f"自動最適化結果 (T_num = {T_num:.6f} 秒)",
            xaxis_title="x", yaxis_title="y",
            yaxis_autorange='reversed'
        )
        st.plotly_chart(fig, use_container_width=True)
