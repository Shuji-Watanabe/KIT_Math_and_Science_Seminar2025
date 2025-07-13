import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import os 
path = os.getcwd()
st.session_state.main_path = path
st.write(path)
if path == '/mount/src/kit_math_and_science_seminar2025':
    location_str = "streamlit_Community_Cloud"
    url = "https://sample-app-simple-linear-regression.streamlit.app/"
else :
    location_str = "local"



# プログラムの実行場所の取得
from mylib import FileProcessing as fp
location_str = fp.location()
st.session_state.location_str = location_str

st.sidebar.markdown(\
"**Updates**\n \
- App 1.0:2025.7.14\n\
")

#====== タイトル
st.header("KIT数理講座2025",divider="rainbow")
"""
このアプリケーションはKIT数理講座2025で使用したプログラムです．
"""

#====== URLとQRコードの表示
st.subheader("Network URL", divider="rainbow")
from mylib import display
from io import BytesIO
if path == '/mount/src/KIT_Math_and_Science_Seminar2025':
    network_url, qr_image = display.display_URL_QRCode(location_str,streamlit_url=url)
else:
    network_url, qr_image = display.display_URL_QRCode(location_str)

## メモリ上に画像を保存
img_bytes = BytesIO()
qr_image.save(img_bytes, format="PNG")  # 画像フォーマットを指定
img_bytes.seek(0)  # ストリームを先頭に移動

left_col, right_col = st.columns([2,1])
with left_col:
    st.write("Network URL:")
    st.code(network_url)
with right_col:
    st.image(img_bytes, caption="2次元コード", use_container_width=True)
    

