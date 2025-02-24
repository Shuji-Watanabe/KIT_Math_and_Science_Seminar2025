import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.interpolate import lagrange, BarycentricInterpolator, CubicSpline, make_interp_spline
from scipy.optimize import minimize
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
# interpolation_method = st.sidebar.selectbox(
#     "補間方法を選択", 
#     ["ラグランジュ補間", "ニュートン補間", "1次スプライン", "2次スプライン", "3次スプライン", "Bスプライン", "n次多項式"]
# )
# --- 制御点の設定（A, B を固定） ---
A = np.array([0.0, 1.0])  # 出発点 (固定)
B = np.array([1.0, 0.0])  # 終点 (固定)

n = num_control  # ユーザーが指定した制御点の数
x_range = np.linspace(0, 1, n)  # x 座標は等間隔
y_init = np.linspace(A[1], B[1], n)  # 初期y座標（直線）
y_init[0], y_init[-1] = A[1], B[1]  # A, B は固定

# # --- 補間関数の適用 ---
# def interpolate(x, y, method):
#     if method == "ラグランジュ補間":
#         return lagrange(x, y)
#     elif method == "ニュートン補間":
#         return BarycentricInterpolator(x, y)
#     elif method == "1次スプライン":
#         return CubicSpline(x, y, bc_type="linear")
#     elif method == "2次スプライン":
#         return CubicSpline(x, y, bc_type="quadratic")
#     elif method == "3次スプライン":
#         return CubicSpline(x, y, bc_type="not-a-knot")
#     elif method == "Bスプライン":
#         return make_interp_spline(x, y, k=3)
#     elif method == "n次多項式":
#         return np.poly1d(np.polyfit(x, y, len(x) - 1))
#     return None

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
    y_fine = np.clip(y_fine, None, A[1])
    dx = x_fine[1] - x_fine[0]
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

    def euler_lagrange_solution():
        """
        最速降下曲線（Brachistochrone）を解析的に求める（単位：m, s）
        - サイクロイド曲線をパラメトリックに表現
        - θ を 0 から θ_max まで変化させて x, y 座標と時間 t を求める
        """
        g = 9.80  # 重力加速度 [m/s²]
        # --- 制御点の設定（A, B を固定） ---
        A = np.array([0.0, 1.0])  # 出発点 (固定)
        B = np.array([1.0, 0.0])  # 終点 (固定)
        # 高さの差 h と横方向の差 w
        h = A[1] - B[1]  # 出発点Aの高さ - 終点Bの高さ [m]
        w = B[0] - A[0]  # 横方向の距離 [m]

        # θ の最大値（終点でのθ）
        theta_max = np.arccos(1 - h)
        # サイクロイドの半径 R を求める
        R = h / (1 - np.cos(theta_max))
        # θの範囲
        theta_vals = np.linspace(0, theta_max, 100)
        # サイクロイドのパラメトリック方程式
        x_raw = R * (theta_vals - np.sin(theta_vals))  # 未スケールの x 座標
        y_sol = A[1] - R * (1 - np.cos(theta_vals))  # [m]
        # x 座標のスケーリング（始点A, 終点Bに合わせる）
        x_sol = A[0] + (x_raw / x_raw[-1]) * w
        # 時間の計算 t = sqrt(2R/g) * θ [s]
        t_sol = np.sqrt(2 * R / g) * theta_vals  # [s]
        return x_sol, y_sol, t_sol


    x_sol, y_sol, t_sol = euler_lagrange_solution()

    if x_sol.size == 0 or y_sol.size == 0:
        st.stop()  # 計算エラーが発生した場合は停止



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
    ax.plot(x_sol, y_sol, 'g-',label="厳密解")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid()

    st.pyplot(fig)

    st.sidebar.write(f"最小移動時間: {res.fun:.3f} 秒")


