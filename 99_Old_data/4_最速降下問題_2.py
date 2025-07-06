
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.interpolate import lagrange, BarycentricInterpolator, CubicSpline, make_interp_spline
from scipy.optimize import minimize
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp


# --- 初期設定 ---
g = 9.80  # 重力加速度 (m/s^2)

# --- Streamlit UI ---
st.title("最速降下曲線")

st.sidebar.header("パラメータ設定")
num_control = st.sidebar.slider("制御点の数 (n)", min_value=3, max_value=100, value=5)
interpolation_method = st.sidebar.selectbox(
    "補間方法を選択", 
    ["3次スプライン", "Bスプライン", "n次多項式"]
)
# --- 制御点の設定（A, B を固定） ---
A = np.array([0.0, 1.0])  # 出発点 (固定)
B = np.array([1.0, 0.0])  # 終点 (固定)

n = num_control  # ユーザーが指定した制御点の数
x_range = np.linspace(0, 1, n)  # x 座標は等間隔
y_init = np.linspace(A[1], B[1], n)  # 初期y座標（直線）
y_init[0], y_init[-1] = A[1], B[1]  # A, B は固定

# --- 補間関数の適用 ---
def interpolate(x, y, method):
    if method == "3次スプライン":
        return CubicSpline(x, y, bc_type="not-a-knot")
    elif method == "Bスプライン":
        return make_interp_spline(x, y, k=3)
    elif method == "n次多項式":
        return np.poly1d(np.polyfit(x, y, len(x) - 1))
    return None


# --- 目的関数（移動時間を最小化） ---
def travel_time(y_points_var):
    y_points = np.hstack(([A[1]],y_points_var , [B[1]]))  # A, B を固定
    interpolator = interpolate(x_range, y_points, interpolation_method)
    x_fine = np.linspace(0, 1, 500)
    y_fine = interpolator(x_fine)
    y_fine[0], y_fine[-1] = A[1], B[1]
    # スタート地点より高くならないよう制限
    # y_fine = np.clip(y_fine, None, A[1])
    dx = x_fine[1:] - x_fine[:-1]
    dy = y_fine[1:] - y_fine[:-1]
    ds = np.sqrt(dx**2 + dy**2)  # 微小弧長
    h = abs(A[1]-y_fine)
    v = np.sqrt(2*g*h)
    v[0] = 0
    v_mean = 0.5*(v[1:]+v[:-1])
    t_total = np.sum(ds / v_mean)  # 時間 = ∫(ds / v)
    return t_total

# --- 最適化途中の可視化用 ---
progress_chart = st.empty()

# --- 最適化実行（スピナー表示 & 途中経過表示） ---
history = []
if st.button("実行"):
    with st.spinner("最適化計算中..."):
        def callback(y):
            y_all = np.hstack(([A[1]], y, [B[1]]))  # A, B を固定
            history.append(y_all)

        res = minimize(travel_time, y_init[1:-1], method="BFGS", callback=callback)


    # 使用例
    theta_T, R, T = 2.412, 0.573, 0.291

    # --- プロット描画 ---
    fig, ax = plt.subplots(figsize=(6, 6))

    # 最適化途中の曲線を描画
    for y_hist in history:
        interpolator_hist = interpolate(x_range, y_hist, interpolation_method)
        y_fine_hist = interpolator_hist(np.linspace(0, 1, 100))
        ax.plot(np.linspace(0, 1, 100), y_fine_hist, color='gray', alpha=0.3)

    # 初期曲線
    interpolator_init = interpolate(x_range, y_init, interpolation_method)
    y_fine_init = interpolator_init(np.linspace(0, 1, 100))
    ax.plot(np.linspace(0, 1, 100), y_fine_init, 'r--', label="初期曲線")

    # 最適化後の曲線
    optimized_y = np.hstack(([A[1]], res.x, [B[1]]))
    interpolator_opt = interpolate(x_range, optimized_y, interpolation_method)
    y_fine_opt = interpolator_opt(np.linspace(0, 1, 100))
    ax.plot(np.linspace(0, 1, 100), y_fine_opt, 'b-', label="最適化曲線")
    ax.scatter(x_range, optimized_y, color='blue',marker="+", label="最適化後制御点", zorder=3)

    # オイラー・ラグランジュ方程式の解
    theta_sol = np.linspace(0,theta_T,100)
    x_sol = R*(theta_sol - np.sin(theta_sol))
    y_sol = -R*(1-np.cos(theta_sol)) + 1
    ax.plot(x_sol, y_sol, 'g-',label="厳密解")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid()

    st.pyplot(fig)

    st.sidebar.write(f"最小移動時間(数値解): {res.fun:.3f} 秒")
    st.sidebar.write(f"最小移動時間(解析解): {2*T:.3f} 秒")
