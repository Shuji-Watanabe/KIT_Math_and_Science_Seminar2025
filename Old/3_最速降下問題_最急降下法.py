import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def compute_time(x, y, g=9.81):
    """移動時間 T を数値積分で計算"""
    y = np.maximum(y, 1e-3)  # 負の値やゼロを回避
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    v = np.sqrt(2 * g * y[:-1])
    T = np.sum(ds / v)
    return T

def gradient_descent(x, y, alpha=0.01, iterations=100):
    """最急降下法で座標を最適化"""
    for _ in range(iterations):
        T = compute_time(x, y)
        grad_y = np.zeros_like(y)
        
        for i in range(1, len(y) - 1):
            y_temp = y.copy()
            epsilon = 1e-5
            y_temp[i] += epsilon
            T_new = compute_time(x, y_temp)
            grad_y[i] = (T_new - T) / epsilon
        
        y -= alpha * grad_y
        y = np.maximum(y, 1e-3)  # 負の値やゼロを防ぐ
    return y

st.title("最速降下曲線 - スプライン最適化")

# パラメータ入力
a = st.number_input("終点 P の x 座標 (a)", value=1.0, format="%.2f")
b = st.number_input("終点 P の y 座標 (b)", value=1.0,format="%.2f")
n = st.slider("通過点の数", min_value=3, max_value=10, value=5)

# 初期点の設定
x = np.linspace(0, a, n)
y = np.linspace(0, b, n)  # 初期値は直線的に配置

# 最適化
y_optimized = gradient_descent(x, y.copy())

# スプライン補間
spline_opt = CubicSpline(x, y_optimized)
x_fine = np.linspace(0, a, 100)
y_fine_opt = spline_opt(x_fine)

# プロット
fig, ax = plt.subplots()
ax.plot(x_fine, y_fine_opt, label="最適化後のスプライン", color="red")
ax.scatter(x, y_optimized, color='green', label="最適化点")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid()
st.pyplot(fig)

# 計算結果の表示
T_optimized = compute_time(x_fine, y_fine_opt)
st.write(f"**最適化後の移動時間 T:** {T_optimized:.4f} 秒")
