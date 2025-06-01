import streamlit as st

def show_footer():
    st.markdown("---")
    st.markdown("""
    <style>
    .header-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .left-column {
        flex: 1;
        text-align: left;
        padding: 10px;
    }
    .middle-column {
        flex: 1;
        text-align: center;
        padding: 10px;
    }
    .right-column {
        flex: 1;
        text-align: right;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="header-container">
        <div class="left-column">
            DỰ ĐOÁN KHẢ NĂNG PHỤC HỒI THẦN KINH Ở BỆNH NHÂN HÔN MÊ SAU NGỪNG TIM SỬ DỤNG CÁC MÔ HÌNH HỌC SÂU<br>
            TP.HCM, tháng 6 năm 2025
        </div>
        <div class="middle-column">
            Nhóm sinh viên thực hiện:<br>
            LƯU HIẾU NGÂN – 21520358<br>
            PHẠM DUY KHÁNH - 21522211
        </div>
        <div class="right-column">
            GIẢNG VIÊN HƯỚNG DẪN<br>
            ThS. DƯƠNG PHI LONG
        </div>
    </div>
    """, unsafe_allow_html=True)