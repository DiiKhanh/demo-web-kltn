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
import scipy.io as sio
from scipy import signal as scipy_signal
import matplotlib.pyplot as plt
import numpy as np

try:
    import librosa
    import torch
    import torch.nn as nn
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    st.warning("⚠️ Librosa hoặc PyTorch không được cài đặt. Mel-spectrogram visualization sẽ không khả dụng.")

from components.header import show_header
from components.footer import show_footer
from components.styles import load_css
from components.tutorial import show_tutorial
from components.result import display_results
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
# Load CSS
st.markdown(load_css(), unsafe_allow_html=True)

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

    # def load_models(self, model_physical_folder):
    #     if not self.load_challenge_models_dynamic:
    #         st.error("❌ Hàm tải model chưa được thiết lập. Vui lòng chọn model hợp lệ.")
    #         return False
    #     try:
    #         if not self.is_loaded:
    #             with st.spinner(f"Đang tải models cho {self.current_model_name} từ {model_physical_folder}..."):
    #                 self.models = self.load_challenge_models_dynamic(model_physical_folder, verbose=1)
    #                 self.is_loaded = True
    #             st.success(f"✅ Models cho {self.current_model_name} từ {model_physical_folder} đã được tải thành công!")
    #         else:
    #             st.info(f"Models cho {self.current_model_name} đã được tải.")
    #         return True
    #     except Exception as e:
    #         st.error(f"❌ Lỗi khi tải models cho {self.current_model_name} từ {model_physical_folder}: {str(e)}")
    #         self.is_loaded = False
    #         return False
    def load_models(self, model_physical_folder):
        if not self.load_challenge_models_dynamic:
            st.error("❌ Hàm tải model chưa được thiết lập. Vui lòng chọn model hợp lệ.")
            return False

        try:
            if not self.is_loaded:
                # Chuyển từ đường dẫn local sang Hugging Face repo_id
                repo_id_map = {
                    "models/densenet121": "your-username/densenet121",
                    "models/resnet50": "your-username/resnet50",
                    "models/convnext": "your-username/convnext",
                    "models/efficentnet-v2-s-72": "your-username/efficientnet-v2-s-72",
                    # 👇 Nếu bạn có thêm improve model thì thêm vào đây
                    # "models/improve/densenet121": "your-username/improve-densenet121"
                }

                repo_id = repo_id_map.get(model_physical_folder)
                if not repo_id:
                    st.error(f"❌ Không tìm thấy repo_id tương ứng với {model_physical_folder}")
                    return False

                with st.spinner(f"📦 Đang tải models cho {self.current_model_name} từ Hugging Face..."):
                    self.models = self.load_challenge_models_dynamic(repo_id, verbose=True)
                    self.is_loaded = True

                st.success(f"✅ Model {self.current_model_name} đã được tải thành công từ Hugging Face!")
            else:
                st.info(f"ℹ️ Model {self.current_model_name} đã được tải.")
            return True

        except Exception as e:
            st.error(f"❌ Lỗi khi tải model {self.current_model_name}: {str(e)}")
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

# IMPORTS REQUIRED (thêm vào đầu file chính):

# BƯỚC 1: Thêm debugging và xử lý lỗi tốt hơn
# Thay thế hàm load_recording_data với version debug này:

def load_recording_data(recording_location):
    """Load EEG recording data from .mat file with debug info"""
    try:
        st.write(f"🔍 Debug: Trying to load from: {recording_location}")
        
        # Kiểm tra file .mat có tồn tại không
        mat_file = recording_location + '.mat'
        hea_file = recording_location + '.hea'
        
        st.write(f"🔍 Debug: Checking files:")
        st.write(f"  - .mat file: {mat_file} - Exists: {os.path.exists(mat_file)}")
        st.write(f"  - .hea file: {hea_file} - Exists: {os.path.exists(hea_file)}")
        
        if not os.path.exists(mat_file):
            st.error(f"❌ .mat file not found: {mat_file}")
            return None, None, None
            
        # Load .mat file
        st.write("📂 Loading .mat file...")
        mat_data = sio.loadmat(mat_file)
        st.write(f"🔍 Debug: .mat file keys: {list(mat_data.keys())}")
        
        # Extract signal data
        recording_data = None
        if 'val' in mat_data:
            recording_data = mat_data['val']
            st.write(f"✅ Found 'val' key with shape: {recording_data.shape}")
        else:
            # Tìm key chứa dữ liệu signal (bỏ qua metadata keys)
            data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            st.write(f"🔍 Debug: Available data keys: {data_keys}")
            
            if data_keys:
                key = data_keys[0]
                recording_data = mat_data[key]
                st.write(f"✅ Using key '{key}' with shape: {recording_data.shape}")
            else:
                st.error("❌ No valid data keys found in .mat file")
                return None, None, None
        
        # Load header file
        channels = []
        sampling_frequency = 250  # default
        
        if os.path.exists(hea_file):
            st.write("📂 Loading .hea file...")
            with open(hea_file, 'r') as f:
                lines = f.readlines()
                st.write(f"🔍 Debug: .hea file has {len(lines)} lines")
                
                if lines:
                    # First line contains basic info
                    first_line = lines[0].strip().split()
                    st.write(f"🔍 Debug: First line: {first_line}")
                    
                    if len(first_line) >= 3:
                        try:
                            sampling_frequency = int(first_line[2])
                            st.write(f"✅ Sampling frequency: {sampling_frequency} Hz")
                        except:
                            st.warning("⚠️ Could not parse sampling frequency, using default 250 Hz")
                    
                    # Following lines contain channel info
                    for i, line in enumerate(lines[1:], 1):
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                channels.append(parts[1])
                    
                    st.write(f"✅ Found {len(channels)} channels: {channels[:5]}..." if len(channels) > 5 else f"✅ Found {len(channels)} channels: {channels}")
        
        # Nếu không có channels từ header, tạo default
        if not channels:
            channels = [f"Channel_{i}" for i in range(recording_data.shape[0])]
            st.write(f"⚠️ Using default channel names: {channels[:5]}..." if len(channels) > 5 else f"⚠️ Using default channel names: {channels}")
            
        st.write(f"✅ Successfully loaded data with shape {recording_data.shape}, {sampling_frequency} Hz, {len(channels)} channels")
        return recording_data, channels, sampling_frequency
        
    except Exception as e:
        st.error(f"❌ Error loading recording data from {recording_location}: {str(e)}")
        st.exception(e)  # Hiển thị full stack trace
        return None, None, None

def create_mel_spectrogram_visualization(patient_folder_path, patient_id, patient_result, channels_to_plot=4, minutes_to_plot=2):
    """Create mel-spectrogram visualization from EEG signals"""
    
    # Kiểm tra thư viện có sẵn không
    if not LIBROSA_AVAILABLE:
        st.error("❌ Không thể tạo mel-spectrogram: Thiếu thư viện librosa hoặc torch")
        return None
    
    try:
        st.write(f"🔍 Debug: Creating mel-spectrogram for patient {patient_id}")
        
        # Tìm file .mat và .hea trong folder
        files = os.listdir(patient_folder_path)
        mat_files = [f for f in files if f.endswith('.mat')]
        
        if not mat_files:
            st.error("No .mat files found for mel-spectrogram")
            return None
            
        # Sử dụng file đầu tiên
        mat_file = mat_files[0]
        base_name = mat_file.replace('.mat', '')
        recording_location = os.path.join(patient_folder_path, base_name)
        
        # Load recording data
        recording_data, channels, sampling_frequency = load_recording_data(recording_location)
        
        if recording_data is None:
            st.error("Could not load recording data for mel-spectrogram")
            return None
        
        # Chọn ngẫu nhiên một channel để tạo mel-spectrogram
        import random
        random_channel_idx = random.randint(0, min(channels_to_plot-1, recording_data.shape[0]-1))
        
        # Lấy signal data từ giữa recording
        samples_per_minute = int(60 * sampling_frequency)
        max_samples = min(int(minutes_to_plot * samples_per_minute), recording_data.shape[1])
        start_idx = max(0, recording_data.shape[1]//2 - max_samples//2)
        end_idx = start_idx + max_samples
        
        signal_data = recording_data[random_channel_idx, start_idx:end_idx]
        channel_name = channels[random_channel_idx] if random_channel_idx < len(channels) else f"Channel_{random_channel_idx}"
        
        st.write(f"🔍 Debug: Using channel {channel_name} (index {random_channel_idx}) with {len(signal_data)} samples")
        
        # Convert to float32 và normalize signal
        signal_data = signal_data.astype(np.float32)
        
        # Normalize signal để tránh lỗi
        if np.std(signal_data) > 0:
            signal_data = (signal_data - np.mean(signal_data)) / np.std(signal_data)
        
        # Tạo mel-spectrogram
        st.write("🎵 Creating mel-spectrogram...")
        spectrograms = librosa.feature.melspectrogram(
            y=signal_data, 
            sr=sampling_frequency, 
            n_mels=224,
            hop_length=512,
            n_fft=2048,
            fmax=sampling_frequency/2  # Thêm fmax để tránh lỗi
        )
        
        # Convert to tensor và normalize
        spectrograms_tensor = torch.from_numpy(spectrograms.astype(np.float32))
        spectrograms_normalized = torch.nn.functional.normalize(spectrograms_tensor, p=2, dim=0)
        
        # Convert to dB scale for visualization
        S_dB = librosa.power_to_db(spectrograms_normalized.numpy(), ref=np.max)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(15, 7))
        
        # Display spectrogram
        img = librosa.display.specshow(
            S_dB, 
            x_axis='time', 
            y_axis='mel',
            sr=sampling_frequency, 
            hop_length=512,  # Thêm hop_length
            ax=ax,
            cmap='viridis'
        )
        
        # Prediction color coding
        pred_color = "🟢 Good" if patient_result['Prediction'] == 'Good' else "🔴 Poor"
        actual_color = "🟢 Good" if patient_result['Actual'] == 'Good' else ("🔴 Poor" if patient_result['Actual'] == 'Poor' else "⚫ Unknown")
        
        ax.set_title(f"Mel-Spectrogram - {channel_name} from Patient {patient_id}\nPrediction: {pred_color} | Actual: {actual_color}", 
                    fontsize=14, pad=20)
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel("Mel Frequency", fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.set_label('Power (dB)', fontsize=12)
        
        plt.tight_layout()
        
        # Thêm thông tin spectrogram
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Mel Bands", "224")
        with col2:
            st.metric("🔊 Sampling Rate", f"{sampling_frequency} Hz")
        with col3:
            st.metric("⏱️ Duration", f"{len(signal_data)/sampling_frequency:.1f} sec")
        with col4:
            st.metric("📈 Shape", f"{spectrograms.shape[0]}x{spectrograms.shape[1]}")
        
        st.write("✅ Mel-spectrogram created successfully")
        return fig
        
    except Exception as e:
        st.error(f"Error creating mel-spectrogram: {str(e)}")
        st.exception(e)
        return None

# BƯỚC 2: Thay thế phần UI visualization với version tự động hiển thị:

def add_eeg_visualization_section(results, all_patient_folders_info, selected_model_display_name, model_type):
    """Separate function for EEG visualization - Auto display without button"""
    
    if not results or len([r for r in results if 'Error' not in r['Prediction']]) == 0:
        st.info("Chạy prediction trước để có thể visualize EEG signals.")
        return
    
    st.markdown("---")
    st.header("📊 EEG Signal Visualization")
    
    # Tạo selectbox để chọn patient cần visualize
    successful_patients = [r['Patient ID'] for r in results if 'Error' not in r['Prediction']]
    
    if not successful_patients:
        st.info("Không có patient nào để visualize (tất cả đều gặp lỗi prediction).")
        return
    
    col_viz1, col_viz2 = st.columns([2, 1])
    
    with col_viz1:
        selected_patient = st.selectbox(
            "Chọn Patient để xem EEG:",
            options=successful_patients,
            help="Chọn bệnh nhân để hiển thị tín hiệu EEG thô",
            key="patient_selector"
        )
    
    with col_viz2:
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            channels_to_plot = st.number_input(
                "Số kênh hiển thị:",
                min_value=1,
                max_value=8,
                value=4,
                help="Số lượng kênh EEG để hiển thị",
                key="channels_input"
            )
        with viz_col2:
            minutes_to_plot = st.number_input(
                "Thời gian (phút):",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.5,
                help="Thời gian tín hiệu EEG để hiển thị",
                key="minutes_input"
            )
    
    # Hiển thị thông tin patient được chọn
    try:
        selected_result = next(r for r in results if r['Patient ID'] == selected_patient)
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Patient ID", selected_patient)
        with col_info2:
            pred_color = "🟢" if selected_result['Prediction'] == 'Good' else "🔴"
            st.metric("Prediction", f"{pred_color} {selected_result['Prediction']}")
        with col_info3:
            actual_color = "🟢" if selected_result['Actual'] == 'Good' else ("🔴" if selected_result['Actual'] == 'Poor' else "⚫")
            st.metric("Actual", f"{actual_color} {selected_result['Actual']}")
    except Exception as e:
        st.error(f"Error displaying patient info: {str(e)}")
        return
    
    # AUTO DISPLAY - Hiển thị ngay lập tức mà không cần button
    st.markdown("### 📈 EEG Signals Display")
    
    try:
        with st.spinner(f"Đang tải và xử lý tín hiệu EEG cho patient {selected_patient}..."):
            
            st.write(f"🔍 Debug: Looking for patient {selected_patient} in {len(all_patient_folders_info)} patient folders")
            
            # Tìm đường dẫn đến data của patient được chọn
            patient_source_path = None
            for patient_id, patient_path in all_patient_folders_info:
                st.write(f"🔍 Debug: Checking patient_id='{patient_id}' vs selected='{selected_patient}'")
                if patient_id == selected_patient:
                    patient_source_path = patient_path
                    st.write(f"✅ Found patient path: {patient_source_path}")
                    break
            
            if not patient_source_path:
                st.error(f"❌ Không tìm thấy đường dẫn data cho patient {selected_patient}")
                st.write("Available patients:")
                for pid, ppath in all_patient_folders_info:
                    st.write(f"  - {pid}: {ppath}")
                return
            
            # Debug: List files in patient folder
            st.write(f"🔍 Debug: Files in patient folder {patient_source_path}:")
            try:
                files = os.listdir(patient_source_path)
                for f in files:
                    st.write(f"  - {f}")
            except Exception as fe:
                st.error(f"Cannot list files in {patient_source_path}: {str(fe)}")
                return
            
            # Gọi hàm visualization
            fig = visualize_eeg_signals_safe(
                patient_source_path,  # direct patient folder path
                selected_patient,
                int(channels_to_plot),
                float(minutes_to_plot)
            )
            
            if fig:
                st.pyplot(fig)
                plt.close(fig)  # Clean up to prevent memory issues

                if LIBROSA_AVAILABLE:
                        st.markdown("### 🎵 Mel-Spectrogram Analysis")
                        
                        # Tạo mel-spectrogram cho channel đầu tiên
                        fig_spec = create_mel_spectrogram_visualization(
                            patient_source_path,
                            selected_patient,
                            selected_result,
                            int(channels_to_plot),
                            float(minutes_to_plot)
                        )
                        
                        if fig_spec:
                            st.pyplot(fig_spec)
                            plt.close(fig_spec)  # Clean up
                        else:
                            st.warning("⚠️ Không thể tạo mel-spectrogram cho patient này")
                else:
                        st.info("💡 Để hiển thị mel-spectrogram, cần cài đặt: `pip install librosa torch`")    
                
                # Thêm thông tin bổ sung
                with st.expander("ℹ️ Thông tin về EEG Visualization", expanded=False):
                    st.markdown(f"""
                    **Thông tin hiển thị:**
                    - **Patient ID**: {selected_patient}
                    - **Prediction**: {selected_result['Prediction']}
                    - **Actual Outcome**: {selected_result['Actual']}
                    - **Số kênh hiển thị**: {int(channels_to_plot)} kênh (được chọn ngẫu nhiên)
                    - **Thời gian**: {float(minutes_to_plot)} phút (từ giữa recording)
                    - **Model sử dụng**: {selected_model_display_name} ({model_type})
                    
                    **Mel-Spectrogram Info:**
                    - **Frequency bins**: 224 mel bands
                    - **Normalization**: L2 normalized
                    - **Display**: Power spectrum in dB scale
                    """)
            else:
                st.error("❌ Không thể tạo visualization cho patient này.")
                
    except Exception as e:
        st.error(f"❌ Lỗi khi tạo EEG visualization: {str(e)}")
        st.exception(e)  # Show full stack trace for debugging

# BƯỚC 3: Hàm visualization an toàn hơn:
def visualize_eeg_signals_safe(patient_folder_path, patient_id, channels_to_plot=4, minutes_to_plot=2):
    """Safe version of EEG visualization with extensive error handling"""
    try:
        st.write(f"🔍 Debug: visualize_eeg_signals_safe called with:")
        st.write(f"  - patient_folder_path: {patient_folder_path}")
        st.write(f"  - patient_id: {patient_id}")
        st.write(f"  - channels_to_plot: {channels_to_plot}")
        st.write(f"  - minutes_to_plot: {minutes_to_plot}")
        
        # Tìm file .mat và .hea trong folder
        files = os.listdir(patient_folder_path)
        mat_files = [f for f in files if f.endswith('.mat')]
        hea_files = [f for f in files if f.endswith('.hea')]
        
        st.write(f"🔍 Debug: Found {len(mat_files)} .mat files and {len(hea_files)} .hea files")
        st.write(f"  - .mat files: {mat_files}")
        st.write(f"  - .hea files: {hea_files}")
        
        if not mat_files:
            st.error(f"No .mat files found in {patient_folder_path}")
            return None
            
        # Sử dụng file đầu tiên
        mat_file = mat_files[0]
        base_name = mat_file.replace('.mat', '')
        recording_location = os.path.join(patient_folder_path, base_name)
        
        st.write(f"🔍 Debug: Using base recording location: {recording_location}")
        
        # Load recording data với debug
        recording_data, channels, sampling_frequency = load_recording_data(recording_location)
        
        if recording_data is None:
            st.error(f"Could not load recording data")
            return None
        
        # Tạo plot đơn giản trước
        st.write(f"🔍 Debug: Creating plot with data shape: {recording_data.shape}")
        
        # Đơn giản hóa - chỉ plot vài channel đầu tiên
        num_channels_to_plot = min(channels_to_plot, recording_data.shape[0], 4)  # Max 4 để an toàn
        
        fig, axes = plt.subplots(num_channels_to_plot, 1, figsize=(12, 8))
        if num_channels_to_plot == 1:
            axes = [axes]
        
        # Plot data đơn giản
        samples_per_minute = int(60 * sampling_frequency)
        max_samples = min(int(minutes_to_plot * samples_per_minute), recording_data.shape[1])
        
        for i in range(num_channels_to_plot):
            # Lấy dữ liệu từ giữa signal
            start_idx = max(0, recording_data.shape[1]//2 - max_samples//2)
            end_idx = start_idx + max_samples
            
            signal_data = recording_data[i, start_idx:end_idx]
            time_axis = np.arange(len(signal_data)) / sampling_frequency / 60  # Convert to minutes
            
            axes[i].plot(time_axis, signal_data, linewidth=0.5)
            channel_name = channels[i] if i < len(channels) else f"Channel_{i}"
            axes[i].set_title(f"{channel_name}", fontsize=12)
            axes[i].set_ylabel("μV", fontsize=10)
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel("Time (minutes)", fontsize=10)
        plt.suptitle(f"EEG Signals - Patient {patient_id}", fontsize=14)
        plt.tight_layout()
        
        st.write("✅ Plot created successfully")
        return fig
        
    except Exception as e:
        st.error(f"Error in visualize_eeg_signals_safe: {str(e)}")
        st.exception(e)
        return None

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

def main():
    show_header()
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

    # col1, col2 = st.columns([2, 1])
    # with col1:
    show_tutorial()
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

# with col2:
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

            # --- EEG VISUALIZATION SECTION (SAFE VERSION) ---
            try:
                add_eeg_visualization_section(results, all_patient_folders_info, selected_model_display_name, "model_type")
            except Exception as e:
                st.error(f"Error in EEG visualization section: {str(e)}")
                st.exception(e)

            if results:
                display_results(results, selected_model_display_name)
            else:
                st.error("❌ Không có kết quả prediction nào!")

    show_footer()

if __name__ == "__main__":
    main()