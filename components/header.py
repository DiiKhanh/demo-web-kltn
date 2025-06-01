import streamlit as st

def show_header():
    st.markdown("""
    <style>
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 20px;
        background-color: #1e1e1e;
        border-bottom: 2px solid #444;
        color: #ffffff;
    }
    .left-section, .right-section {
        display: flex;
        align-items: center;
    }
    .left-section img, .right-section img {
        height: 80px;
        margin-right: 15px;
        margin-left: 15px;
        filter: brightness(0.9);
    }
    .left-text, .right-text {
        font-size: 18px;
        font-weight: bold;
        line-height: 1.4;
        color: #ffffff;
    }
    </style>
    <div class="header-container">
        <div class="left-section">
            <img src="https://www.uit.edu.vn/sites/vi/files/images/Logos/Logo_UIT_Web_Transparent.png" alt="Logo Trường">
            <div class="left-text">
                TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN - ĐHQG-HCM<br>
                KHOA HỆ THỐNG THÔNG TIN
            </div>
        </div>
        <div class="right-section">
            <div class="right-text">
                DỰ ĐOÁN KHẢ NĂNG PHỤC HỒI THẦN KINH Ở BỆNH NHÂN HÔN MÊ SAU NGỪNG TIM<br>
                SỬ DỤNG CÁC MÔ HÌNH HỌC SÂU
            </div>
            <img src="https://cdn-icons-png.flaticon.com/512/9851/9851782.png" alt="Logo Đề tài">
        </div>
    </div>
    """, unsafe_allow_html=True)