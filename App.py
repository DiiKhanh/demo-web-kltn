import streamlit as st
import os
import sys
import tempfile
import shutil
import zipfile
import time
from pathlib import Path
import pandas as pd
import plotly.express as px
import importlib  # Required for dynamic imports
import plotly.graph_objects as go
import numpy as np
import scipy.io

# Cấu hình đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
# Modified: Use current_dir since team_code_densenet.py is in the same directory as App.py
sys.path.append(current_dir)

# --- CORRECTED: Ensure helper_code functions are imported ---
try:
    from helper_code import load_text_file, get_variable, get_cpc
except ImportError as e:
    st.error(f"Không thể import helper_code: {e}. Đảm bảo helper_code.py tồn tại trong {current_dir}.")
    st.stop()

# --- Dynamic Importer with Debug Output ---
def get_model_functions(model_module_name):
    """
    Dynamically imports load_challenge_models and run_challenge_models
    from the specified model module.
    """
    # st.write(f"Debug: Attempting to import module '{model_module_name}'")
    # st.write(f"Debug: Current sys.path: {sys.path}")
    try:
        module = importlib.import_module(model_module_name)
        st.write(f"Debug: Successfully imported module '{model_module_name}'")
        load_models_func = getattr(module, 'load_challenge_models')
        run_models_func = getattr(module, 'run_challenge_models')
        st.write(f"Debug: Found functions in '{model_module_name}'")
        return load_models_func, run_models_func
    except ImportError as e:
        st.error(f"❌ Không thể import module: {model_module_name}. Error: {str(e)}. Đảm bảo file {model_module_name}.py tồn tại trong {current_dir}.")
        return None, None
    except AttributeError as e:
        st.error(f"❌ Module {model_module_name} thiếu hàm 'load_challenge_models' hoặc 'run_challenge_models'. Error: {str(e)}")
        return None, None
    except Exception as e:
        st.error(f"❌ Lỗi không xác định khi import từ {model_module_name}: {str(e)}")
        return None, None

# Cấu hình trang
st.set_page_config(
    page_title="EEG Prediction App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-result {
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: bold;
    margin: 1rem 0;
}
.good-result {
    background-color: #d4edda;
    color: #155724;
    border: 2px solid #28a745;
}
.poor-result {
    background-color: #f8d7da;
    color: #721c24;
    border: 2px solid #dc3545;
}
.debug-section {
    border: 1px solid #e9ecef;
    border-radius: 5px;
    padding: 1rem;
    background-color: #f8f9fa;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

class EEGPredictor:
    def __init__(self):
        self.models = None
        self.is_loaded = False
        self.current_model_name = None
        self.load_challenge_models_dynamic = None
        self.run_challenge_models_dynamic = None

    def set_model_functions(self, model_name, load_func, run_func):
        if self.current_model_name != model_name:
            self.models = None
            self.is_loaded = False
            self.current_model_name = model_name
        self.load_challenge_models_dynamic = load_func
        self.run_challenge_models_dynamic = run_func

    def load_models(self, model_physical_folder):
        if not self.load_challenge_models_dynamic:
            st.error("❌ Hàm tải model chưa được thiết lập. Vui lòng chọn model hợp lệ.")
            return False
        try:
            if not self.is_loaded:
                with st.spinner(f"Đang tải models cho {self.current_model_name} từ {model_physical_folder}..."):
                    self.models = self.load_challenge_models_dynamic(model_physical_folder, verbose=1)
                    self.is_loaded = True
                st.success(f"✅ Models cho {self.current_model_name} từ {model_physical_folder} đã được tải thành công!")
            else:
                st.info(f"Models cho {self.current_model_name} đã được tải.")
            return True
        except Exception as e:
            st.error(f"❌ Lỗi khi tải models cho {self.current_model_name} từ {model_physical_folder}: {str(e)}")
            self.is_loaded = False
            return False

    def predict_single_patient(self, temp_data_folder, patient_id, model_physical_folder):
        if not self.run_challenge_models_dynamic:
            st.error("❌ Hàm predict model chưa được thiết lập. Vui lòng chọn model hợp lệ.")
            return None, None, None
        try:
            if not self.is_loaded:
                st.warning(f"Models cho {self.current_model_name} chưa được tải. Đang thử tải...")
                if not self.load_models(model_physical_folder):
                    return None, None, None

            patient_folder = os.path.join(temp_data_folder, patient_id)
            if not os.path.exists(patient_folder):
                st.error(f"Không tìm thấy folder patient: {patient_id}")
                return None, None, None
            files_in_folder = os.listdir(patient_folder)
            hea_files = [f for f in files_in_folder if f.endswith('.hea')]
            mat_files = [f for f in files_in_folder if f.endswith('.mat')]
            if not hea_files or not mat_files:
                st.error(f"Thiếu file .hea hoặc .mat trong folder {patient_id}")
                return None, None, None
            metadata_file = os.path.join(patient_folder, f"{patient_id}.txt")
            actual_outcome = None
            if os.path.exists(metadata_file):
                try:
                    meta_data = load_text_file(metadata_file)
                    actual_outcome = get_variable(meta_data, 'Outcome', str) if meta_data else None
                except Exception as e_meta:
                    st.warning(f"Không thể đọc outcome từ metadata file {metadata_file}: {e_meta}")
                    pass
            else:
                with open(metadata_file, 'w') as f:
                    f.write(f"Patient: {patient_id}\n")
                    f.write("Age: Unknown\n")
                    f.write("Sex: Unknown\n")
                    f.write("Outcome: Unknown\n")

            with st.spinner(f"Đang predict cho patient {patient_id} sử dụng {self.current_model_name}..."):
                outcome_binary, outcome_probability = self.run_challenge_models_dynamic(
                    self.models, temp_data_folder, patient_id, verbose=0
                )
            return outcome_binary, outcome_probability, actual_outcome
        except Exception as e:
            st.error(f"Lỗi khi predict patient {patient_id} với {self.current_model_name}: {str(e)}")
            return None, None, None

def debug_folder_structure(base_path, level=0, max_level=3):
    debug_info = []
    if level > max_level:
        return debug_info
    try:
        items = os.listdir(base_path)
        for item in items:
            item_path = os.path.join(base_path, item)
            indent = "  " * level
            if os.path.isdir(item_path):
                debug_info.append(f"{indent}📁 {item}/")
                try:
                    files = os.listdir(item_path)
                    hea_files = [f for f in files if f.endswith('.hea')]
                    mat_files = [f for f in files if f.endswith('.mat')]
                    if hea_files or mat_files:
                        debug_info.append(f"{indent}  → .hea files: {len(hea_files)}, .mat files: {len(mat_files)}")
                    if level < max_level:
                        sub_debug = debug_folder_structure(item_path, level + 1, max_level)
                        debug_info.extend(sub_debug)
                except PermissionError:
                    debug_info.append(f"{indent}  → (Permission denied)")
            else:
                file_ext = os.path.splitext(item)[1]
                if file_ext in ['.hea', '.mat', '.txt']:
                    debug_info.append(f"{indent}📄 {item}")
    except Exception as e:
        debug_info.append(f"{indent}❌ Error reading {base_path}: {str(e)}")
    return debug_info

def extract_uploaded_files(uploaded_files, temp_dir):
    extracted_folders_map = {}
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_path = os.path.join(temp_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if file_name.endswith('.zip'):
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                    extracted_folders_map[temp_dir] = True
                    st.success(f"✅ Đã giải nén: {file_name} vào {temp_dir}")
            except Exception as e:
                st.error(f"❌ Lỗi khi giải nén {file_name}: {str(e)}")
    return list(extracted_folders_map.keys())

def find_patient_folders(base_path, debug_mode=False):
    patient_folders_dict = {}
    if debug_mode:
        st.info(f"🔍 Scanning directory: {base_path}")
        st.markdown("**📁 Folder Structure (during find_patient_folders):**")
        debug_info = debug_folder_structure(base_path, max_level=2)
        if debug_info:
            for line in debug_info[:30]:
                st.text(line)
            if len(debug_info) > 30:
                st.text(f"... and {len(debug_info) - 30} more items")
    for root, dirs, files in os.walk(base_path):
        hea_files = [f for f in files if f.endswith('.hea')]
        mat_files = [f for f in files if f.endswith('.mat')]
        if debug_mode and (hea_files or mat_files or dirs):
            relative_path = os.path.relpath(root, base_path)
        if hea_files and mat_files:
            folder_name = os.path.basename(root)
            if folder_name not in patient_folders_dict:
                if root != base_path or (root == base_path and not any(os.path.isdir(os.path.join(root, d)) for d in dirs if d != "prediction_input_data")):
                    patient_folders_dict[folder_name] = root
                    if debug_mode:
                        st.success(f"✅ Tentatively found patient: {folder_name} at {root}")
    patient_folders = list(patient_folders_dict.items())
    if debug_mode and not patient_folders:
        st.warning(f"No patient folders found directly in {base_path} or its subdirectories.")
    return patient_folders

# VISUALIZE
def parse_metadata_file(file_path):
    """Parse patient metadata from .txt file"""
    metadata = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for line in content.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
        
        return metadata
    except Exception as e:
        st.error(f"Lỗi khi đọc metadata file {file_path}: {str(e)}")
        return {}

def parse_hea_file(hea_file_path):
    """Parse WFDB header file to extract signal information"""
    try:
        with open(hea_file_path, 'r') as f:
            lines = f.readlines()
        
        header_info = {}
        if lines:
            # First line contains record info
            record_line = lines[0].strip().split()
            header_info['record_name'] = record_line[0]
            header_info['n_signals'] = int(record_line[1])
            header_info['sampling_freq'] = float(record_line[2])
            header_info['n_samples'] = int(record_line[3])
            
            # Parse signal information
            signals = []
            for i in range(1, min(len(lines), header_info['n_signals'] + 1)):
                signal_line = lines[i].strip().split()
                if len(signal_line) >= 2:
                    signal_info = {
                        'filename': signal_line[0],
                        'format': signal_line[1],
                        'gain': float(signal_line[2]) if len(signal_line) > 2 else 1.0,
                        'baseline': int(signal_line[3]) if len(signal_line) > 3 else 0,
                        'units': signal_line[4] if len(signal_line) > 4 else 'mV',
                        'description': ' '.join(signal_line[8:]) if len(signal_line) > 8 else f'Signal {i}'
                    }
                    signals.append(signal_info)
            
            header_info['signals'] = signals
        
        return header_info
    except Exception as e:
        st.error(f"Lỗi khi đọc header file {hea_file_path}: {str(e)}")
        return {}

def load_mat_file(mat_file_path):
    """Load EEG signal data from MAT file"""
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        
        # Find the actual data variable (exclude metadata keys)
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        
        if data_keys:
            # Usually the signal data is stored in 'val' or similar key
            main_key = data_keys[0]
            signal_data = mat_data[main_key]
            
            # Ensure data is 2D (channels x samples)
            if signal_data.ndim == 1:
                signal_data = signal_data.reshape(1, -1)
            
            return signal_data
        else:
            st.error("Không tìm thấy dữ liệu signal trong file MAT")
            return None
            
    except Exception as e:
        st.error(f"Lỗi khi đọc MAT file {mat_file_path}: {str(e)}")
        return None

def visualize_patient_metadata(metadata_dict, patient_id):
    """Display patient metadata in a formatted table"""
    st.subheader(f"📋 Thông tin bệnh nhân: {patient_id}")
    
    if metadata_dict:
        # Create a formatted dataframe for display
        metadata_df = pd.DataFrame([
            {"Thông tin": "Patient ID", "Giá trị": metadata_dict.get('Patient', 'N/A')},
            {"Thông tin": "Bệnh viện", "Giá trị": metadata_dict.get('Hospital', 'N/A')},
            {"Thông tin": "Tuổi", "Giá trị": metadata_dict.get('Age', 'N/A')},
            {"Thông tin": "Giới tính", "Giá trị": metadata_dict.get('Sex', 'N/A')},
            {"Thông tin": "ROSC (phút)", "Giá trị": metadata_dict.get('ROSC', 'N/A')},
            {"Thông tin": "OHCA", "Giá trị": metadata_dict.get('OHCA', 'N/A')},
            {"Thông tin": "Shockable Rhythm", "Giá trị": metadata_dict.get('Shockable Rhythm', 'N/A')},
            {"Thông tin": "TTM (°C)", "Giá trị": metadata_dict.get('TTM', 'N/A')},
            {"Thông tin": "Kết quả", "Giá trị": metadata_dict.get('Outcome', 'N/A')},
            {"Thông tin": "CPC Score", "Giá trị": metadata_dict.get('CPC', 'N/A')}
        ])
        
        # Color code the outcome
        def highlight_outcome(val):
            if 'Kết quả' in str(val):
                return 'background-color: #721c24'
            elif 'Good' in str(val):
                return 'background-color: #d4edda; color: #155724'
            elif 'Poor' in str(val):
                return 'background-color: #f8d7da; color: #721c24'
            return ''
        
        styled_df = metadata_df.style.applymap(highlight_outcome)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Additional insights
        col1, col2, col3 = st.columns(3)
        with col1:
            outcome = metadata_dict.get('Outcome', 'Unknown')
            if outcome == 'Good':
                st.success(f"✅ Kết quả: {outcome}")
            elif outcome == 'Poor':
                st.error(f"❌ Kết quả: {outcome}")
            else:
                st.info(f"❓ Kết quả: {outcome}")
        
        with col2:
            age = metadata_dict.get('Age', 'N/A')
            st.metric("Tuổi", age)
        
        with col3:
            cpc = metadata_dict.get('CPC', 'N/A')
            st.metric("CPC Score", cpc)
    else:
        st.warning("Không có thông tin metadata cho bệnh nhân này")

def visualize_eeg_signals(signal_data, header_info, patient_id, max_duration=60):
    """Visualize EEG signals with interactive plots"""
    st.subheader(f"🧠 Tín hiệu EEG - Bệnh nhân: {patient_id}")
    
    if signal_data is None or header_info is None:
        st.error("Không thể tải dữ liệu EEG")
        return
    
    # Get basic info
    n_channels, n_samples = signal_data.shape
    sampling_freq = header_info.get('sampling_freq', 250.0)  # Default 250 Hz
    duration_seconds = n_samples / sampling_freq
    
    # Display signal information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Số kênh", n_channels)
    with col2:
        st.metric("Tần số lấy mẫu", f"{sampling_freq:.1f} Hz")
    with col3:
        st.metric("Số mẫu", f"{n_samples:,}")
    with col4:
        st.metric("Thời lượng", f"{duration_seconds/60:.1f} phút")
    
    # Time axis
    time_axis = np.arange(n_samples) / sampling_freq
    
    # Limit display duration for performance
    if duration_seconds > max_duration:
        st.info(f"⚠️ Hiển thị {max_duration} giây đầu tiên (tổng: {duration_seconds/60:.1f} phút)")
        max_samples = int(max_duration * sampling_freq)
        signal_data = signal_data[:, :max_samples]
        time_axis = time_axis[:max_samples]
        n_samples = max_samples
    
    # Channel selection
    channel_names = []
    if 'signals' in header_info and header_info['signals']:
        channel_names = [sig.get('description', f'Channel {i+1}') for i, sig in enumerate(header_info['signals'])]
    else:
        channel_names = [f'Channel {i+1}' for i in range(n_channels)]
    
    # Interactive channel selection
    selected_channels = st.multiselect(
        "Chọn kênh EEG để hiển thị:",
        options=list(range(n_channels)),
        default=list(range(min(4, n_channels))),  # Default to first 4 channels
        format_func=lambda x: channel_names[x] if x < len(channel_names) else f'Channel {x+1}'
    )
    
    if not selected_channels:
        st.warning("Vui lòng chọn ít nhất một kênh để hiển thị")
        return
    
    # Plotting options
    col1, col2 = st.columns(2)
    with col1:
        plot_type = st.selectbox("Kiểu hiển thị:", ["Tách riêng", "Chồng lên nhau"])
    with col2:
        amplitude_scale = st.slider("Độ phóng đại biên độ:", 0.1, 5.0, 1.0, 0.1)
    
    # Create plots
    if plot_type == "Tách riêng":
        # Separate subplots for each channel
        fig = go.Figure()
        
        for i, ch_idx in enumerate(selected_channels):
            # Scale and offset signals for better visualization
            offset = i * (np.std(signal_data[ch_idx]) * 4)
            scaled_signal = signal_data[ch_idx] * amplitude_scale + offset
            
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=scaled_signal,
                mode='lines',
                name=channel_names[ch_idx] if ch_idx < len(channel_names) else f'Channel {ch_idx+1}',
                line=dict(width=1)
            ))
        
        fig.update_layout(
            title=f"Tín hiệu EEG - {patient_id}",
            xaxis_title="Thời gian (giây)",
            yaxis_title="Biên độ (μV)",
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
    else:
        # Overlapped signals
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        for i, ch_idx in enumerate(selected_channels):
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=signal_data[ch_idx] * amplitude_scale,
                mode='lines',
                name=channel_names[ch_idx] if ch_idx < len(channel_names) else f'Channel {ch_idx+1}',
                line=dict(width=1, color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            title=f"Tín hiệu EEG (Chồng lên) - {patient_id}",
            xaxis_title="Thời gian (giây)",
            yaxis_title="Biên độ (μV)",
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Signal statistics
    st.subheader("📊 Thống kê tín hiệu")
    stats_data = []
    for ch_idx in selected_channels:
        channel_data = signal_data[ch_idx]
        stats_data.append({
            'Kênh': channel_names[ch_idx] if ch_idx < len(channel_names) else f'Channel {ch_idx+1}',
            'Trung bình': f"{np.mean(channel_data):.2f} μV",
            'Độ lệch chuẩn': f"{np.std(channel_data):.2f} μV",
            'Min': f"{np.min(channel_data):.2f} μV",
            'Max': f"{np.max(channel_data):.2f} μV",
            'RMS': f"{np.sqrt(np.mean(channel_data**2)):.2f} μV"
        })
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Power spectral density (optional)
    if st.checkbox("Hiển thị phân tích tần số (PSD)"):
        st.subheader("📈 Mật độ phổ công suất (PSD)")
        
        fig_psd = go.Figure()
        for ch_idx in selected_channels[:4]:  # Limit to 4 channels for PSD
            # Compute PSD using Welch's method
            from scipy import signal as scipy_signal
            freqs, psd = scipy_signal.welch(signal_data[ch_idx], sampling_freq, nperseg=1024)
            
            # Only show frequencies up to 50 Hz (typical EEG range)
            freq_mask = freqs <= 50
            freqs = freqs[freq_mask]
            psd = psd[freq_mask]
            
            fig_psd.add_trace(go.Scatter(
                x=freqs,
                y=10 * np.log10(psd),  # Convert to dB
                mode='lines',
                name=channel_names[ch_idx] if ch_idx < len(channel_names) else f'Channel {ch_idx+1}',
                line=dict(width=2)
            ))
        
        fig_psd.update_layout(
            title="Mật độ phổ công suất",
            xaxis_title="Tần số (Hz)",
            yaxis_title="PSD (dB/Hz)",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_psd, use_container_width=True)

def visualize_patient_data(patient_folder_path, patient_id):
    """Main function to visualize all patient data"""
    st.markdown("---")
    st.header(f"🔍 Dữ liệu chi tiết - Bệnh nhân: {patient_id}")
    
    # Find files
    files_in_folder = os.listdir(patient_folder_path)
    txt_files = [f for f in files_in_folder if f.endswith('.txt')]
    hea_files = [f for f in files_in_folder if f.endswith('.hea')]
    mat_files = [f for f in files_in_folder if f.endswith('.mat')]
    
    # Display file information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Files .txt", len(txt_files))
    with col2:
        st.metric("Files .hea", len(hea_files))
    with col3:
        st.metric("Files .mat", len(mat_files))
    
    # Visualize metadata
    if txt_files:
        txt_file_path = os.path.join(patient_folder_path, txt_files[0])
        metadata = parse_metadata_file(txt_file_path)
        visualize_patient_metadata(metadata, patient_id)
    
    # Visualize EEG signals
    if hea_files and mat_files:
        # Use the first pair of hea/mat files
        hea_file_path = os.path.join(patient_folder_path, hea_files[0])
        mat_file_path = os.path.join(patient_folder_path, mat_files[0])
        
        # Parse header and load signal data
        header_info = parse_hea_file(hea_file_path)
        signal_data = load_mat_file(mat_file_path)
        
        if signal_data is not None:
            visualize_eeg_signals(signal_data, header_info, patient_id)
        else:
            st.error("Không thể tải dữ liệu tín hiệu EEG")
    else:
        st.warning("Không tìm thấy files .hea hoặc .mat để hiển thị tín hiệu EEG")
# VISUALIZE
def main():
    # st.markdown('<h1 class="main-header">🧠 EEG Prediction System</h1>', unsafe_allow_html=True)
    # HEADER
    st.markdown(
    """
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
    """,
    unsafe_allow_html=True
    )
    # HEADER
    st.sidebar.header("⚙️ Cấu hình")
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False, help="Show detailed folder structure and debugging info")

    # --- CORRECTED: Model configuration with accurate module name for EfficientNet ---
    model_config = {
        "DenseNet-121": {"module": "team_code_densenet", "path": "models/densenet121"},
        "ResNet-50": {"module": "team_code_resnet50", "path": "models/resnet50"},
        "ConvNeXt": {"module": "team_code_convnext", "path": "models/convnext"},
        "EfficientNet-V2-S": {"module": "team_code_efficient", "path": "models/efficentnet-v2-s-72"} # Corrected module name
    }

    selected_model_display_name = st.sidebar.selectbox(
        "Chọn Model:",
        options=list(model_config.keys()),
        help="Chọn model đã train để sử dụng cho prediction."
    )

    selected_model_module_name = model_config[selected_model_display_name]["module"]
    selected_model_physical_path = model_config[selected_model_display_name]["path"]

    if 'predictor' not in st.session_state:
        st.session_state.predictor = EEGPredictor()

    load_fn, run_fn = get_model_functions(selected_model_module_name)
    if load_fn and run_fn:
        st.session_state.predictor.set_model_functions(selected_model_display_name, load_fn, run_fn)
    else:
        st.sidebar.error(f"Không thể tải các hàm cho model {selected_model_display_name}. Kiểm tra tên module '{selected_model_module_name}.py' và đảm bảo file tồn tại.")

    if st.sidebar.button("🔄 Tải Models", key="load_models_button"):
        if st.session_state.predictor.load_challenge_models_dynamic:
            st.session_state.predictor.load_models(selected_model_physical_path)
        else:
            st.sidebar.error("Hàm tải model chưa được thiết lập do lỗi import. Vui lòng chọn model hợp lệ và kiểm tra thông báo lỗi.")

    col1, col2 = st.columns([2, 1])
    with col1:
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
        uploaded_files = st.file_uploader(
            "Chọn file ZIP chứa dữ liệu EEG",
            accept_multiple_files=True,
            type=['zip'],
            help="Upload file ZIP chứa các folder bệnh nhân với file .hea và .mat"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        if uploaded_files:
            st.success(f"✅ Đã upload {len(uploaded_files)} file(s)")
            file_info = [{"Tên File": f.name, "Kích thước": f"{f.size / (1024*1024):.2f} MB", "Loại": "ZIP Archive"} for f in uploaded_files]
            st.dataframe(pd.DataFrame(file_info), use_container_width=True)

    with col2:
        st.header("🎯 Prediction")
        if st.button("🚀 Bắt đầu Predict", type="primary", use_container_width=True, key="predict_button"):
            if not uploaded_files:
                st.warning("⚠️ Vui lòng upload files EEG trước!")
                return

            if not st.session_state.predictor.load_challenge_models_dynamic or \
                not st.session_state.predictor.run_challenge_models_dynamic:
                st.error("❌ Model functions không được tải đúng cách. Vui lòng kiểm tra lựa chọn model và thông báo lỗi ở sidebar.")
                return

            if not st.session_state.predictor.is_loaded:
                st.warning(f"⚠️ Models cho {st.session_state.predictor.current_model_name} chưa được tải! Đang thử tải...")
                if not st.session_state.predictor.load_models(selected_model_physical_path):
                    st.error("Không thể tải models. Prediction bị hủy.")
                    return

            with tempfile.TemporaryDirectory() as temp_dir:
                if debug_mode: st.info(f"🔧 Debug: Using temp directory: {temp_dir}")
                # st.info("📦 Đang xử lý files upload...") # Can be noisy
                base_extraction_path = temp_dir
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(base_extraction_path, uploaded_file.name)
                    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
                    if uploaded_file.name.endswith('.zip'):
                        try:
                            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                                zip_ref.extractall(base_extraction_path)
                                # st.success(f"✅ Đã giải nén: {uploaded_file.name} vào {base_extraction_path}")
                        except Exception as e:
                            st.error(f"❌ Lỗi khi giải nén {uploaded_file.name}: {str(e)}")
                            continue
                    else: st.warning(f"Skipping non-ZIP file: {uploaded_file.name}")

                if debug_mode:
                    st.markdown(f"### 🐛 Debug: Structure of temp_dir after extraction: {base_extraction_path}")
                    debug_tree = debug_folder_structure(base_extraction_path, max_level=2)
                    for line in debug_tree: st.text(line)

                st.info("🔍 Đang tìm patient folders...")
                all_patient_folders_info = find_patient_folders(base_extraction_path, debug_mode=debug_mode)

                if not all_patient_folders_info:
                    st.error("❌ Không tìm thấy patient data hợp lệ trong files upload.")
                    if debug_mode:
                        st.markdown("### 🐛 Debug Help for No Patients Found:")
                        st.markdown(f"""
                        **Kiểm tra các vấn đề sau:**
                        1. File ZIP có thực sự chứa các **folder con** không? (ví dụ: `patient_ID_1/`, `patient_ID_2/`)
                        2. Mỗi folder con (ví dụ: `patient_ID_1/`) có chứa cả file `.hea` và `.mat` không?
                        3. Tên file có đúng định dạng không?
                        4. Cấu trúc thư mục có khớp với hướng dẫn không?
                        **Cấu trúc thư mục được quét trong `{base_extraction_path}`:**
                        Ví dụ: `{base_extraction_path}/0391/0391.hea` và `{base_extraction_path}/0391/0391.mat`
                        """)
                    return

                st.success(f"✅ Tìm thấy {len(all_patient_folders_info)} patient(s).")
                if debug_mode:
                    st.markdown("### 📋 Found Patients for Prediction:")
                    for patient_id, patient_original_path in all_patient_folders_info:
                        st.text(f"  👤 {patient_id} (source: {patient_original_path})")

                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                prediction_input_dir = os.path.join(temp_dir, "prediction_input_data")
                os.makedirs(prediction_input_dir, exist_ok=True)

                for i, (patient_id, patient_original_path) in enumerate(all_patient_folders_info):
                    progress_bar.progress((i + 1) / len(all_patient_folders_info))
                    status_text.text(f"🔄 Đang predict cho Patient {patient_id} ({i+1}/{len(all_patient_folders_info)})")
                    temp_patient_run_folder = os.path.join(prediction_input_dir, patient_id)
                    os.makedirs(temp_patient_run_folder, exist_ok=True)
                    try:
                        for item_name in os.listdir(patient_original_path):
                            src_item = os.path.join(patient_original_path, item_name)
                            dst_item = os.path.join(temp_patient_run_folder, item_name)
                            if os.path.isfile(src_item): shutil.copy2(src_item, dst_item)
                        if debug_mode:
                            copied_files = os.listdir(temp_patient_run_folder)
                            # st.text(f"  Copied {len(copied_files)} files to {temp_patient_run_folder} for patient {patient_id}")
                    except Exception as e:
                        st.error(f"Error copying files for {patient_id}: {str(e)}")
                        results.append({'Patient ID': patient_id, 'Prediction': 'Error - File Prep', 'Actual': "N/A"})
                        continue
                    
                    outcome_binary, outcome_prob, actual_outcome = st.session_state.predictor.predict_single_patient(
                        prediction_input_dir, patient_id, selected_model_physical_path
                    )

                    if outcome_binary is not None:
                        results.append({
                            'Patient ID': patient_id,
                            'Prediction': 'Good' if outcome_binary == 0 else 'Poor',
                            # 'Probability': f"{outcome_prob:.4f}" if outcome_prob is not None else "N/A",
                            'Actual': actual_outcome if actual_outcome else "Unknown"
                        })
                    else:
                        results.append({
                            'Patient ID': patient_id,
                            'Prediction': 'Error - Prediction Failed',
                            # 'Probability': "N/A",
                            'Actual': actual_outcome if actual_outcome else "N/A" # Keep actual if read
                        })

                progress_bar.empty()
                status_text.empty()

                if results:
                    st.header("📊 Kết Quả Prediction")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    st.subheader("📈 Phân phối Kết quả Prediction")
                    prediction_counts = results_df['Prediction'].value_counts().reset_index()
                    prediction_counts.columns = ['Prediction', 'Count']
                    color_map = {'Good': '#28a745', 'Poor': '#dc3545', 'Error - Prediction Failed': '#ffc107', 'Error - File Prep': '#6c757d'}
                    for pred_type in prediction_counts['Prediction']:
                        if pred_type not in color_map: color_map[pred_type] = '#007bff'
                    fig = px.bar(prediction_counts, x='Prediction', y='Count', color='Prediction', color_discrete_map=color_map, title="Số lượng theo từng loại Prediction", labels={'Count': 'Số lượng bệnh nhân', 'Prediction': 'Kết quả Dự đoán'})
                    fig.update_layout(xaxis_title="Kết quả Dự đoán", yaxis_title="Số lượng bệnh nhân")
                    st.plotly_chart(fig, use_container_width=True)
                    st.subheader("💡 Kết quả chi tiết từng Patient")
                    for _, row in results_df.iterrows():
                        patient_id, prediction = row['Patient ID'], row['Prediction']
                        patient_folder = os.path.join(prediction_input_dir, patient_id)
                        st.write(f"Debug: Checking patient folder: {patient_folder}")
                        
                        if os.path.exists(patient_folder):
                            st.write(f"Debug: Found patient folder: {patient_folder}, contents: {os.listdir(patient_folder)}")
                            try:
                                visualize_patient_data(patient_folder, patient_id)
                            except Exception as e:
                                st.error(f"Lỗi khi hiển thị dữ liệu cho bệnh nhân {patient_id}: {e}")
                        else:
                            st.warning(f"Không tìm thấy thư mục dữ liệu cho bệnh nhân {patient_id}")
                        # In kết quả prediction
                        if prediction == 'Good': 
                            st.markdown(f'''<div class="prediction-result good-result">👤 {patient_id}: {prediction}</div>''', unsafe_allow_html=True)
                        elif prediction == 'Poor': 
                            st.markdown(f'''<div class="prediction-result poor-result">👤 {patient_id}: {prediction}</div>''', unsafe_allow_html=True)
                        else:
                            st.error(f"👤 {patient_id}: {prediction}")
                    good_count = sum(1 for r in results if r['Prediction'] == 'Good')
                    poor_count = sum(1 for r in results if r['Prediction'] == 'Poor')
                    error_count = sum(1 for r in results if 'Error' in r['Prediction'])
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1: st.metric("Total Patients", len(results))
                    with col_stat2: st.metric("Good Outcomes", good_count)
                    with col_stat3: st.metric("Poor Outcomes", poor_count)
                    with col_stat4: st.metric("Errors", error_count)
                    csv_data = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Download Results (CSV)", data=csv_data, file_name=f"eeg_predictions_{selected_model_display_name.replace('/','_').replace(' ','_')}_{int(time.time())}.csv", mime="text/csv")
                else:
                    st.error("❌ Không có kết quả prediction nào!")

    st.markdown("---")
    st.markdown(
    """
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
    """,
    unsafe_allow_html=True
    )
    st.markdown(
    """
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
    """,
    unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()