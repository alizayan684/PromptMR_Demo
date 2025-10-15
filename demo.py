import streamlit as st
import os
import subprocess
import tempfile
import scipy.io as sio
import numpy as np
from PIL import Image

st.title("PromptMR MRI Reconstruction Web App")
# Get the directory where this script is located.
try:
    # The directory of the currently running script.
    script_dir = os.path.dirname(os.path.realpath(__file__))
except NameError:
    # Fallback for environments where __file__ is not defined
    script_dir = os.getcwd()

repo_dir = os.path.abspath(os.path.join(script_dir, "PromptMR"))

uploaded_file = st.file_uploader("Upload .mat file for k-space", type=["mat"])

task = st.selectbox("Task", ["Cine", "Mapping"])

if uploaded_file is not None:
    # Check if the local PromptMR directory exists
    if not os.path.isdir(repo_dir):
        st.error("Error: The 'PromptMR' directory was not found.")
        st.info(f"Expected location: {repo_dir}")
        st.error("Please ensure the 'PromptMR' directory is in the same folder as the 'demo.py' script.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:

        input_root = os.path.join(tmpdir, "input")
        input_dir = os.path.join(input_root, "MultiCoil")
        set_dir = os.path.join(input_dir, task, "ValidationSet") # Assume Cine for Both, adjust if needed
        
        os.makedirs(set_dir, exist_ok=True)
        mat_path = os.path.join(set_dir, uploaded_file.name)
        with open(mat_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Assume pretrained_models directory exists in repo_dir with the ckpt file
        # If not, manually place pretrained_models/promptmr-16cascades-epoch=11-step=258576.ckpt in the repo_dir
        model_path = os.path.join(repo_dir, "pretrained_models", "promptmr-16cascades-epoch=11-step=258576.ckpt")
        if not os.path.exists(model_path):
            st.error(f"Error: Pretrained model not found at {model_path}. Please ensure it's downloaded and placed in the correct directory.")
            st.stop()
        script_path = os.path.join(repo_dir, "promptmr_examples", "cmrxrecon", "run_pretrained_promptmr_cmrxrecon_inference_from_matlab_data.py")
        inference_dir = os.path.dirname(script_path)

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        command = [
            "python",
            script_path,
            "--input", input_dir,
            "--output", output_dir,
            "--model_path", model_path,
            "--evaluate_set", "ValidationSet",
            "--task", task,
            "--batch_size", "4",
            "--num_workers", "2",
            "--center_crop",
            "--num_cascades", "16"
        ]
        
        try:
            st.info(f"Running inference command: {' '.join(command)}")
            result = subprocess.run(command, cwd=inference_dir, capture_output=True, text=True, check=True)
            st.success("Inference completed successfully!")
            st.code(result.stdout, language='bash')
        except subprocess.CalledProcessError as e:
            st.error(f"Error running the model: {e}\nStderr: {e.stderr}\nStdout: {e.stdout}")
        
        # Find output .mat files in output_dir
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".mat"):
                    output_mat_path = os.path.join(root, file)
                    data = sio.loadmat(output_mat_path)
                    
                    # Assume 'reconstruction_rss' or 'reconstruction_esc' key for the image
                    img_key = next((k for k in data.keys() if 'reconstruction' in k.lower()), None)
                    if img_key:
                        img = data[img_key]
                    else:
                        # Fallback to assume complex image and take magnitude
                        possible_keys = [k for k in data if not k.startswith('__')]
                        if possible_keys:
                            img = np.abs(data[possible_keys[0]])
                        else:
                            st.error("No suitable image data found in .mat")
                            break
                    
                    # Handle multi-dimensional images: take a representative slice
                    if img.ndim > 2:
                        # Assume dimensions: [slices, height, width] or similar; take middle slice
                        slice_idx = img.shape[0] // 2
                        img = img[slice_idx]
                    if img.ndim > 2:
                        # If still >2, take first phase or channel
                        img = img[0]
                    
                    # Normalize to 0-255
                    img_min, img_max = img.min(), img.max()
                    if img_max > img_min:
                        img = (img - img_min) / (img_max - img_min) * 255
                    img = img.astype(np.uint8)
                    
                    # Display
                    pil_img = Image.fromarray(img, mode='L')  # Grayscale
                    st.image(pil_img, caption="Reconstructed MRI Image", use_column_width=True)
                    break
            else:
                continue
            break
        else:
            st.error("No output .mat file found.")