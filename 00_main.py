import os
import streamlit as st
from io import BytesIO
import qrcode

#====== 実行パスの取得
path = os.getcwd()
st.session_state.main_path = path

#====== 実行環境の判定
if path == '/mount/src/kit_math_and_science_seminar2025':
    location_str = "streamlit_Community_Cloud"
    url = "https://sample-app-simple-linear-regression.streamlit.app/"
else:
    location_str = "local"
st.session_state.location_str = location_str

#====== アップデート情報
st.sidebar.markdown(
    "**Updates**\n"
    "- App 1.0:2025.7.14\n"
)

#====== タイトルと説明
st.header("KIT数理講座2025", divider="rainbow")
"""
このアプリケーションはKIT数理講座2025で使用したプログラムです．
"""

#====== QRコードとURLの表示
st.subheader("Network URL", divider="rainbow")

def generate_qr_code(url: str):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")

# URLとQRコード生成
if location_str == "streamlit_Community_Cloud":
    network_url = url
else:
    import socket
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    network_url = f"http://{ip_address}:8501"

qr_image = generate_qr_code(network_url)

# メモリ上に画像を保存
img_bytes = BytesIO()
qr_image.save(img_bytes, format="PNG")
img_bytes.seek(0)

# 表示
left_col, right_col = st.columns([2, 1])
with left_col:
    st.write("Network URL:")
    st.code(network_url)
with right_col:
    st.image(img_bytes, caption="2次元コード", use_container_width=True)
