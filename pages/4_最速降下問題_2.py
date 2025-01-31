import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# 最速降下法に基づく時間最小化のための目的関数の勾配
def compute_gradient(points):
    gradients = []
    for i in range(1, len(points) - 1):
        # 各点における勾配（時間最小化のための近似）
        grad = 2 * (points[i] - points[i-1]) + 2 * (points[i] - points[i+1])
        gradients.append(grad)
    gradients = np.array(gradients)
    return gradients

# 勾配降下法
def gradient_descent(gradient_func, start_points, learning_rate=0.01, max_iter=1000):
    path = [start_points.copy()]
    current_points = start_points.copy()
    
    for _ in range(max_iter):
        # 勾配計算
        gradient = gradient_func(current_points)
        
        # 勾配に沿って更新
        next_points = current_points[1:-1] - learning_rate * gradient
        
        # 先端と末端は固定する
        updated_points = np.vstack([current_points[0], next_points, current_points[-1]])
        path.append(updated_points)
        
        # 収束判定
        if np.linalg.norm(next_points - current_points[1:-1]) < 1e-6:
            break
            
        current_points = updated_points
    
    return np.array(path)

# Streamlitアプリケーション
st.title("重力による最速降下曲線の最急降下法")

# ユーザー設定
a = st.number_input("到着点のx座標 (a)", value=5.0)
b = st.number_input("到着点のy座標 (b)", value=5.0)
n = st.slider("曲線Cの通過点数 (n)", 2, 20, 5)
learning_rate = st.slider("学習率", 0.001, 0.1, 0.01)
max_iter = st.slider("最大反復回数", 100, 10000, 1000)

# 出発点は原点 (0, 0)
start_point = np.array([0, 0])

# 到着点 (a, b)
end_point = np.array([a, b])

# n個の点を直線的に生成して最初の点を設定
points = np.linspace(start_point, end_point, n)

# 最急降下法で最速降下曲線を求める
path = gradient_descent(compute_gradient, points, learning_rate, max_iter)

# 結果の描画
fig, ax = plt.subplots()
for i in range(len(path)):
    ax.plot(path[i][:, 0], path[i][:, 1], marker="o", label=f"ステップ {i}")

ax.scatter(0, 0, color='red', label="出発点 (0, 0)", zorder=5)
ax.scatter(a, b, color='blue', label="到着点 (a, b)", zorder=5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("最速降下曲線")
ax.legend()
st.pyplot(fig)
