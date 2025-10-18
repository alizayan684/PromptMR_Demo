import streamlit as st
import os
import subprocess
import tempfile
import scipy.io as sio
import numpy as np
from PIL import Image

# Page configuration for wide layout
st.set_page_config(
    page_title="PromptMR - MRI Reconstruction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
        border-bottom: 3px solid #3b82f6;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Image container styling */
    .image-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .image-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .image-label {
        font-size: 1rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Control panel styling */
    .control-panel {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .status-success {
        background-color: #10b981;
        color: white;
    }
    
    .status-processing {
        background-color: #f59e0b;
        color: white;
    }
    
    .status-error {
        background-color: #ef4444;
        color: white;
    }
    
    /* Grid layout improvements */
    .stColumn > div {
        height: 100%;
    }
    
    /* Info boxes */
    .info-box {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">🏥 PromptMR MRI Reconstruction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Medical Image Processing & Multi-View Visualization</p>', unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.markdown("### ⚙️ Configuration Panel")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📁 Upload K-Space Data",
        type=["mat"],
        help="Upload a .mat file containing MRI k-space data"
    )
    
    task = st.selectbox(
        "🎯 Reconstruction Task",
        ["Cine", "Mapping"],
        help="Select the type of MRI reconstruction task"
    )
    
    st.markdown("---")
    st.markdown("### 🔧 Advanced Settings")
    
    batch_size = st.slider("Batch Size", 1, 8, 4, help="Number of samples processed simultaneously")
    num_workers = st.slider("Worker Threads", 1, 8, 2, help="Number of parallel data loading threads")
    num_cascades = st.slider("Model Cascades", 8, 24, 16, help="Number of reconstruction cascades")
    
    center_crop = st.checkbox("Center Crop", value=True, help="Apply center cropping to output")
    
    st.markdown("---")
    st.markdown("### 📊 Processing Info")
    st.info("**Status:** Ready for upload")
    
    if uploaded_file:
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.success(f"**File:** {uploaded_file.name}")
        st.metric("File Size", f"{file_size:.2f} MB")

# Initialize session state for storing images
if 'reconstructed_images' not in st.session_state:
    st.session_state.reconstructed_images = {}

# Main processing logic
if uploaded_file is not None:
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        script_dir = os.getcwd()

    repo_dir = os.path.abspath(os.path.join(script_dir, "PromptMR"))

    if not os.path.isdir(repo_dir):
        st.error("❌ **Error:** The 'PromptMR' directory was not found.")
        st.info(f"📂 Expected location: `{repo_dir}`")
        st.warning("Please ensure the 'PromptMR' directory is in the same folder as the 'demo.py' script.")
        st.stop()

    # Processing button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button("🚀 Start Reconstruction", use_container_width=True)
    
    if process_button:
        with st.spinner('🔄 Processing MRI reconstruction...'):
            with tempfile.TemporaryDirectory() as tmpdir:
                input_root = os.path.join(tmpdir, "input")
                input_dir = os.path.join(input_root, "MultiCoil")
                set_dir = os.path.join(input_dir, task, "ValidationSet")
                
                os.makedirs(set_dir, exist_ok=True)
                mat_path = os.path.join(set_dir, uploaded_file.name)
                with open(mat_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                output_dir = os.path.join(tmpdir, "output")
                os.makedirs(output_dir, exist_ok=True)
                
                model_path = os.path.join(repo_dir, "pretrained_models", "promptmr-16cascades-epoch=11-step=258576.ckpt")
                if not os.path.exists(model_path):
                    st.error(f"❌ Pretrained model not found at `{model_path}`")
                    st.stop()
                
                script_path = os.path.join(repo_dir, "promptmr_examples", "cmrxrecon", "run_pretrained_promptmr_cmrxrecon_inference_from_matlab_data.py")
                inference_dir = os.path.dirname(script_path)

                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                
                command = [
                    "python", script_path,
                    "--input", input_dir,
                    "--output", output_dir,
                    "--model_path", model_path,
                    "--evaluate_set", "ValidationSet",
                    "--task", task,
                    "--batch_size", str(batch_size),
                    "--num_workers", str(num_workers),
                    "--num_cascades", str(num_cascades)
                ]
                
                if center_crop:
                    command.append("--center_crop")
                
                try:
                    with st.expander("📝 View Command Details"):
                        st.code(' '.join(command), language='bash')
                    
                    result = subprocess.run(command, cwd=inference_dir, capture_output=True, text=True, check=True)
                    st.success("✅ Reconstruction completed successfully!")
                    
                    with st.expander("📋 View Processing Log"):
                        st.code(result.stdout, language='bash')
                    
                    # Process output files
                    output_images = []
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            if file.endswith(".mat"):
                                output_mat_path = os.path.join(root, file)
                                data = sio.loadmat(output_mat_path)
                                
                                img_key = next((k for k in data.keys() if 'reconstruction' in k.lower()), None)
                                if img_key:
                                    img_data = data[img_key]
                                else:
                                    possible_keys = [k for k in data if not k.startswith('__')]
                                    if possible_keys:
                                        img_data = np.abs(data[possible_keys[0]])
                                    else:
                                        continue
                                
                                # Extract multiple slices/phases if available
                                if img_data.ndim > 2:
                                    # Extract 4 representative slices
                                    total_slices = img_data.shape[0]
                                    indices = [
                                        0,  # First
                                        total_slices // 3,  # Early middle
                                        2 * total_slices // 3,  # Late middle
                                        total_slices - 1  # Last
                                    ]
                                    
                                    for idx, slice_idx in enumerate(indices):
                                        img = img_data[slice_idx]
                                        if img.ndim > 2:
                                            img = img[0]
                                        
                                        # Normalize
                                        img_min, img_max = img.min(), img.max()
                                        if img_max > img_min:
                                            img = (img - img_min) / (img_max - img_min) * 255
                                        img = img.astype(np.uint8)
                                        
                                        output_images.append({
                                            'image': Image.fromarray(img, mode='L'),
                                            'label': f'Slice {slice_idx + 1}/{total_slices}',
                                            'index': slice_idx
                                        })
                                else:
                                    # Single 2D image
                                    img_min, img_max = img_data.min(), img_data.max()
                                    if img_max > img_min:
                                        img_data = (img_data - img_min) / (img_max - img_min) * 255
                                    img = img_data.astype(np.uint8)
                                    output_images.append({
                                        'image': Image.fromarray(img, mode='L'),
                                        'label': 'Reconstructed Image',
                                        'index': 0
                                    })
                    
                    st.session_state.reconstructed_images = output_images[:4]  # Store up to 4 images
                    
                except subprocess.CalledProcessError as e:
                    st.error(f"❌ **Error during reconstruction:**")
                    st.code(e.stderr, language='text')

# Display reconstructed images in 2x2 grid
if st.session_state.reconstructed_images:
    st.markdown("---")
    st.markdown("## 🖼️ Multi-View Reconstruction Results")
    st.markdown("*Compare different slices or phases of the reconstructed MRI data*")
    
    # Create 2x2 grid
    images = st.session_state.reconstructed_images
    
    # Top row
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        if len(images) > 0:
            st.markdown(f'<div class="image-label">🔍 View 1: {images[0]["label"]}</div>', unsafe_allow_html=True)
            st.image(images[0]['image'], use_container_width=True, caption=f"Resolution: {images[0]['image'].size[0]}×{images[0]['image'].size[1]}")
    
    with col2:
        if len(images) > 1:
            st.markdown(f'<div class="image-label">🔍 View 2: {images[1]["label"]}</div>', unsafe_allow_html=True)
            st.image(images[1]['image'], use_container_width=True, caption=f"Resolution: {images[1]['image'].size[0]}×{images[1]['image'].size[1]}")
    
    # Bottom row
    col3, col4 = st.columns(2, gap="large")
    
    with col3:
        if len(images) > 2:
            st.markdown(f'<div class="image-label">🔍 View 3: {images[2]["label"]}</div>', unsafe_allow_html=True)
            st.image(images[2]['image'], use_container_width=True, caption=f"Resolution: {images[2]['image'].size[0]}×{images[2]['image'].size[1]}")
    
    with col4:
        if len(images) > 3:
            st.markdown(f'<div class="image-label">🔍 View 4: {images[3]["label"]}</div>', unsafe_allow_html=True)
            st.image(images[3]['image'], use_container_width=True, caption=f"Resolution: {images[3]['image'].size[0]}×{images[3]['image'].size[1]}")
    
    # Download options
    st.markdown("---")
    st.markdown("### 💾 Export Options")
    col1, col2, col3, col4 = st.columns(4)
    
    for idx, (col, img_data) in enumerate(zip([col1, col2, col3, col4], images)):
        if idx < len(images):
            with col:
                # Convert PIL image to bytes for download
                from io import BytesIO
                buf = BytesIO()
                img_data['image'].save(buf, format='PNG')
                st.download_button(
                    label=f"⬇️ Download View {idx + 1}",
                    data=buf.getvalue(),
                    file_name=f"reconstruction_view_{idx + 1}.png",
                    mime="image/png",
                    use_container_width=True
                )

else:
    # Placeholder when no images
    st.markdown("---")
    st.markdown("## 🖼️ Multi-View Visualization")
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="image-label">🔍 View 1</div>', unsafe_allow_html=True)
        st.info("Upload a file and run reconstruction to display results here")
        
        st.markdown('<div class="image-label">🔍 View 3</div>', unsafe_allow_html=True)
        st.info("Multiple slices will be displayed for comparison")
    
    with col2:
        st.markdown('<div class="image-label">🔍 View 2</div>', unsafe_allow_html=True)
        st.info("High-quality MRI reconstructions from k-space data")
        
        st.markdown('<div class="image-label">🔍 View 4</div>', unsafe_allow_html=True)
        st.info("Professional medical imaging visualization")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem 0;'>
    <p><strong>PromptMR</strong> - Advanced MRI Reconstruction Platform</p>
    <p style='font-size: 0.875rem;'>Powered by Deep Learning | Professional Medical Imaging</p>
</div>
""", unsafe_allow_html=True)