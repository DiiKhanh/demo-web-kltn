import streamlit as st

def show_tutorial():
    st.header("📁 Upload EEG Data")
    st.markdown('<div>', unsafe_allow_html=True)
    st.markdown("**📋 Hướng dẫn upload:**")
    st.markdown("""
    - Upload file ZIP.
    - File ZIP phải chứa các **folder đặt tên theo ID bệnh nhân** (ví dụ: 0391, 1234, patient_001).
    - Mỗi folder bệnh nhân phải chứa:
        - File `.hea` (header file)
        - File `.mat` (data file)
        - Tùy chọn: File `.txt` (metadata bệnh nhân, nếu có sẽ đọc Outcome thực tế)
    - **Cấu trúc ZIP được khuyến nghị:**
    ```
        your_data.zip
        ├── 0391/
        │   ├── 0391.hea
        │   ├── 0391.mat
        │   └── (0391.txt)
        ├── 1234/
        │   ├── 1234.hea
        │   ├── 1234.mat
        │   └── (1234.txt)
        └── ...
    ```
    """)