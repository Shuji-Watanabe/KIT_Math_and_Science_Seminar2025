import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve

st.title("最速経路探索２")

st.header("初期パラメータの設定",divider="violet")
with st.expander("設定の確認・変更") :
    #a, b = 0.60, 0.4
    """ ##### 原点と座標軸
    原点＝始点，x軸の向き：右向き正，y軸の向き：下向き正
    """
    """ ##### 終点の位置の設定"""
    input_cols = st.columns([1]*4)
    with input_cols[0] :
        a    = st.number_input("終点の $x$ 座標 $\\rm [m]$"
                                , min_value=float(0)
                                ,value=np.pi
                                ,format='%.4f'
                                ,step=0.0001)
        a = float(a)
    with input_cols[1] :
        b    = st.number_input("終点の $y$ 座標 $\\rm [m]$"
                                ,min_value=float(0)
                                ,value=float(2)
                                ,format='%.4f'
                                ,step=0.0001)  
        b    = float(b)
    f""" ##### 分割数
    $x=0$ から $x={a:.3f}$ の間を $N$ 分割します．（初期設定は等分割）
    """
    if not "N" in st.session_state:
        st.session_state.N = 10
    N_col = st.columns([1,1,1])
    with N_col[0]:
        N = st.number_input("分割数 N", min_value=1, max_value=20, value=10, step=1)
    
    with N_col[2]:
        if st.session_state.N != N:
            if st.button("分割数$N$を更新しますか?"):
                st.session_state.N = N
                del st.session_state.saved_datasets
                with N_col[1]:
                    st.success("更新完了")
            else:   
                with N_col[1]:
                    st.write(f'$N={st.session_state.N}$を使用')
    # """ ##### 最急降下法用のパラメータ"""
    # input_cols = st.columns([1]*4)
    # with input_cols[0] :
    #     max_iter    = st.number_input("最大更新回数"
    #                             , min_value=1
    #                             , max_value=50000
    #                             ,value=20000)
    # with input_cols[1] :
    #     alpha    = st.number_input("$\\alpha$"
    #                             , min_value=0.0001
    #                             , max_value=5
    #                             , value=20000)
        
    #                                           gamma_nodes=1.0,
    #                                           kappa=2, delta=3,
    #                                           alpha=3, beta=0.0001,
    #                                           momentum=0.8,
    #                                           max_iter=20000, tol=1e-9
    # """
    # $$ 
    # \\mathbf{y}_{i,k+1} = \\mathbf{y}_{i,k} 
    # - m \\cdot v_{i,k} +\\frac{\\alpha}{1+\\beta \\cdot k}
    # \\cdot \\frac{1}{1+e^{ - \\kappa\\cdot \\big(x_{i}/x_0 -\\delta\\big) }}
    # \\cdot 	\\frac{\\partial T}{\\partial y_{i}} \\Big|_{\mathbf{x}=\mathbf{x}_{k-1},\mathbf{y}=\mathbf{y}_{k-1} }
    # $$
    # """
    
    # with input_cols[0] :
           
# ---------------------------------
# コア関数定義
# ---------------------------------
def nonuniform_nodes(a, N, gamma_nodes=1):
    u = np.linspace(0, 1, N+1)
    return a * u**gamma_nodes

def travel_time_manual(yk, xk, g=9.80665):
    """
    以前の手動評価関数 travel_time_manual を Simpson 法版に置き換えました。
    """
    return travel_time_numeric(yk, xk, g=g, interp_method= 'cubic')

from scipy.interpolate import interp1d, make_interp_spline
from scipy.optimize import fsolve
def travel_time_numeric(yk, xk, g=9.80665,
                        interp_method='linear',
                        degree=3,  # for polynomial/B-spline
                        num_points=500,  # 分割数
                        eps=1e-8):
    """
    yk, xk: 各ノードの y, x 配列 (長さ N+1)
    interp_method: 'linear', 'quadratic', 'cubic', 'bspline'
    degree: 多項式次数（quadratic=2, cubic=3）
    num_points: x軸方向の評価点数
    """
    # 補完関数の作成
    if interp_method == 'linear':
        f = interp1d(xk, yk, kind='linear')
    elif interp_method in ('quadratic', 'cubic'):
        k = 2 if interp_method=='quadratic' else 3
        f = make_interp_spline(xk, yk, k=k)
    elif interp_method == 'bspline':
        # Bスプライン（3次）の例
        f = make_interp_spline(xk, yk, k=degree)
    else:
        raise ValueError(f"Unknown interp_method: {interp_method}")

    # 評価点生成
    xs = np.linspace(xk[0], xk[-1], num_points)
    ys = f(xs)

    # ds/v を区間ごとに足し合わせ
    dx = np.diff(xs)
    dy = np.diff(ys)
    ds = np.hypot(dx, dy)
    y_mid = np.maximum((ys[:-1] + ys[1:]) / 2, eps)
    v_mid = np.sqrt(2 * g * y_mid)
    t = np.sum(ds / v_mid)
    return t




def compute_gradient_simpson(yk, xk, eps=1e-6):
    N = len(xk) - 1
    grad = np.zeros_like(yk)
    for i in range(1, N):
        ykp, ykm = yk.copy(), yk.copy()
        ykp[i] += eps; ykm[i] -= eps
        grad[i] = (travel_time_numeric(ykp, xk) - travel_time_numeric(ykm, xk)) / (2 * eps)
    return grad

def steepest_descent_simpson_sigmoid_momentum(a, b, N,yk=None,
                                              gamma_nodes=1.0,
                                              kappa=2, delta=3,
                                              alpha=3, beta=0.0001,
                                              momentum=0.8,
                                              max_iter=20000, tol=1e-9):
    if delta is None:
        delta = 2 - 0.01 * N
    xk = nonuniform_nodes(a, N, gamma_nodes)
    if yk is None:
        yk = b * (xk / a)
    else:
        yk = np.array(yk, dtype=float)  
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
xk = nonuniform_nodes(a, N, gamma_nodes=1)
# セッションステート初期化
if 'saved_datasets' not in st.session_state:
    st.session_state.saved_datasets = []
col2_0 = st.columns([2,1])

with col2_0[0]:
    # モード選択
    mode = st.radio("モード選択", ["手動最適化", "自動最適化"],horizontal=True)

with col2_0[1]:
    # パスワード入力
    code_input = st.text_input("解答表示のパスワード")

if mode == "手動最適化":
    st.markdown("### 各通過点の y 座標入力")
    Col_num = 5
    Input_cols = st.columns( Col_num )
    # 保存済みデータ選択
    data_col = st.columns([1,1,1,1,2])
    with data_col[-1]:
        if st.session_state.saved_datasets:
            idx = st.selectbox("保存済みデータを選択", list(range(len(st.session_state.saved_datasets))),
                            format_func=lambda i: f"データ {i}", key="select_dataset")
            selected_y = st.session_state.saved_datasets[idx]['y']
        else:
            selected_y = None
            
    yk = []
    for j in range(len(xk)):
        if j == 0:
            Input_cols = st.columns( Col_num )
            with Input_cols[0]:
                yj = st.number_input(
                    f"$y_{{{j}}}$", value=float(0)
                    ,min_value=float(0),max_value=float(0), key=f"manual_y_{0}"
                )
            yk.append(0.0)
        elif j == len(xk)-1:
            # with Input_cols[ j%Col_num]:
            Input_cols = st.columns( Col_num )
            with Input_cols[ 0]:
                yj = st.number_input(
                    f"$y_{{{j}}}$", value=b,
                    format="%.4f",step=0.0001, min_value=b,max_value=b, key=f"manual_y_{j}"
                )
            yk.append(b)
        else:
            if j==1 :
                Input_cols = st.columns( Col_num )
            default = selected_y[j] if selected_y is not None else b*(xk[j]/a)
            if N <= 20 :
                with Input_cols[ (j-1)%Col_num]:
                    yj = st.number_input(
                        f"$y_{{{j}}}$", value=float(default),
                        format="%.4f", step=0.0001, key=f"manual_y_{j}"
                    )
                yk.append(yj)
            else :
                yk.append(default)
    # 移動時間計算（等加速度モデル）
    T_manual = travel_time_manual(yk, xk)
        
    with data_col[0]:
        if st.button("保　　存"):
            st.session_state.saved_datasets.append({'y': yk.copy(), 'T': T_manual})
            with data_col[3]:
                st.success("処理完了")
    with data_col[1]:
        if st.button("リセット"):
            for j in range(1, len(xk)-1):
                st.session_state.pop(f"manual_y_{j}", None)
            del st.session_state.saved_datasets
            with data_col[3]:
                st.success("処理完了")
    with data_col[2]:
        if st.button("更　　新"):
            with data_col[3]:
                st.success("処理完了")
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
                title=f"手動解 (T = {T_manual:.4f} 秒)",
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
        # st.write("保存して")
        yk = []
        for j in range(len(xk)):
            if j == 0:
                yk.append(0.0)
            elif j == len(xk)-1:
                yk.append(b)
            else:
                yk.append( b*(xk[j]/a))
        T_manual = travel_time_manual(yk, xk)
        st.session_state.saved_datasets.append({'y': yk, 'T': T_manual})
elif mode == "自動最適化":
    st.markdown("### 自動最適化（手動最適化の結果を初期値に使用）")

    # 最小Tの保存データを初期値として
    if st.session_state.saved_datasets:
        best = min(st.session_state.saved_datasets, key=lambda d: d['T'])
        yk_init = best['y']
    else:
        yk_init = [0.0] + [b*(x/a) for x in xk[1:-1]] + [b]

    # st.write("初期解（保存データから）:", [f"{y:.4f}" for y in yk_init])

    if st.button("Start Optimization"):
        xk_opt, yk_opt = steepest_descent_simpson_sigmoid_momentum(a, b, N, yk=yk_init)
        # T_num = travel_time_simpson(yk_opt, xk_opt)
        T_num = travel_time_numeric( yk_opt
                                    ,xk_opt
                                    ,interp_method='cubic')
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
        if code_input == "5963":
            xc, yc = exact_cycloid(a, b)
            fig.add_trace(go.Scatter(
                x=xc, y=yc, mode='lines',
                name='厳密解 (Cycloid)',
                line=dict(color='orange', dash='dash')
            ))
        fig.update_layout(
            title=f"自動最適化結果 (T_num = {T_num:.4f} 秒)",
            xaxis_title="x", yaxis_title="y",
            yaxis_autorange='reversed'
        )
        st.plotly_chart(fig, use_container_width=True)
        # 結果表示
        # 手動評価 (加速度ベース)
        T_manual = travel_time_manual(yk_init, xk)
        # 自動評価 (Simpson)
        T_auto = travel_time_numeric(yk_opt, xk)
        # 厳密解評価
        x_c, y_c = exact_cycloid(a, b)
        R = b / (1 - np.cos(solve_theta(a, b)))
        T_exact = np.pi * np.sqrt(R / 9.81)
        """#### 移動時間"""
        Result_col = st.columns(3)
        with Result_col[0]:
            st.write(f"手動最適化経路: {T_manual:.4f} 秒")
        with Result_col[1]:
            st.write(f"自動最適化経路: {T_auto:.4f} 秒")
        with Result_col[2]:
            st.write(f"サイクロイド:{T_exact:.4f} 秒")