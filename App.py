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

from components.header import show_header
from components.footer import show_footer
from components.styles import load_css
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
                        # patient_id, prediction, probability = row['Patient ID'], row['Prediction'], row['Probability']
                        # if prediction == 'Good': st.markdown(f'''<div class="prediction-result good-result">👤 {patient_id}: {prediction} (Prob: {probability})</div>''', unsafe_allow_html=True)
                        # elif prediction == 'Poor': st.markdown(f'''<div class="prediction-result poor-result">👤 {patient_id}: {prediction} (Prob: {probability})</div>''', unsafe_allow_html=True)
                        patient_id, prediction = row['Patient ID'], row['Prediction']
                        if prediction == 'Good': 
                            st.markdown(f'''<div class="prediction-result good-result">👤 {patient_id}: {prediction}</div>''', unsafe_allow_html=True)
                        elif prediction == 'Poor': 
                            st.markdown(f'''<div class="prediction-result poor-result">👤 {patient_id}: {prediction}</div>''', unsafe_allow_html=True)
                        else: st.error(f"👤 {patient_id}: {prediction}")
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

    show_footer()

if __name__ == "__main__":
    main()