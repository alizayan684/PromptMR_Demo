import streamlit as st
import os
import subprocess
import tempfile
import scipy.io as sio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64

# Page configuration for wide layout
st.set_page_config(
    page_title="AI-Powered - MRI Reconstruction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for compact, professional styling
st.markdown("""
<style>
    /* Remove padding and margins */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    /* Compact title styling */
    .main-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin: 0;
        padding: 0.5rem 0;
        border-bottom: 2px solid #3b82f6;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        margin: 0.25rem 0 0.5rem 0;
    }

    /* Compact image container styling */
    .image-container {
        background: #ffffff;
        border-radius: 8px;
        padding: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        height: 100%;
    }

    .image-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.25rem;
        padding-bottom: 0.25rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Compact sidebar */
    section[data-testid="stSidebar"] {
        width: 280px !important;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    /* Reduce spacing */
    .stMarkdown {
        margin-bottom: 0.25rem;
    }

    h3 {
        font-size: 1rem;
        margin: 0.5rem 0 0.25rem 0;
    }

    hr {
        margin: 0.5rem 0;
    }

    /* Compact buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        font-size: 0.9rem;
        border-radius: 6px;
        margin: 0.25rem 0;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }

    /* Compact file uploader */
    .stFileUploader {
        padding: 0;
    }

    /* Compact selectbox and sliders */
    .stSelectbox, .stSlider {
        margin-bottom: 0.5rem;
    }

    /* Compact info boxes */
    .element-container div[data-testid="stNotification"] {
        padding: 0.5rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
    }

    /* Compact metric */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.8rem;
    }

    /* Reduce column gaps */
    [data-testid="column"] {
        padding: 0.25rem;
    }

    /* Compact expander */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        padding: 0.5rem;
    }

    /* Image captions */
    .stImage > div {
        font-size: 0.75rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 0.5rem 0;
        font-size: 0.8rem;
        margin-top: 0.5rem;
        border-top: 1px solid #e2e8f0;
    }

    /* Remove extra spacing from images */
    .stImage {
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)


# Function to create placeholder images
def create_placeholder_image(view_num, width=400, height=400):
    """Create a professional medical imaging placeholder"""
    # Create image with gradient background
    img = Image.new('RGB', (width, height), color='#1e293b')
    draw = ImageDraw.Draw(img)

    # Add gradient effect
    for y in range(height):
        shade = int(30 + (y / height) * 40)
        color = (shade, shade, shade + 10)
        draw.line([(0, y), (width, y)], fill=color)

    # Add grid pattern for medical imaging feel
    grid_spacing = 40
    for x in range(0, width, grid_spacing):
        draw.line([(x, 0), (x, height)], fill='#334155', width=1)
    for y in range(0, height, grid_spacing):
        draw.line([(0, y), (width, y)], fill='#334155', width=1)

    # Add central cross (targeting reticle)
    center_x, center_y = width // 2, height // 2
    cross_size = 60
    draw.line([(center_x - cross_size, center_y), (center_x + cross_size, center_y)],
              fill='#3b82f6', width=2)
    draw.line([(center_x, center_y - cross_size), (center_x, center_y + cross_size)],
              fill='#3b82f6', width=2)

    # Add corner markers
    marker_size = 20
    marker_offset = 20
    # Top-left
    draw.line([(marker_offset, marker_offset), (marker_offset + marker_size, marker_offset)],
              fill='#3b82f6', width=2)
    draw.line([(marker_offset, marker_offset), (marker_offset, marker_offset + marker_size)],
              fill='#3b82f6', width=2)
    # Top-right
    draw.line([(width - marker_offset - marker_size, marker_offset), (width - marker_offset, marker_offset)],
              fill='#3b82f6', width=2)
    draw.line([(width - marker_offset, marker_offset), (width - marker_offset, marker_offset + marker_size)],
              fill='#3b82f6', width=2)
    # Bottom-left
    draw.line([(marker_offset, height - marker_offset), (marker_offset + marker_size, height - marker_offset)],
              fill='#3b82f6', width=2)
    draw.line([(marker_offset, height - marker_offset - marker_size), (marker_offset, height - marker_offset)],
              fill='#3b82f6', width=2)
    # Bottom-right
    draw.line([(width - marker_offset - marker_size, height - marker_offset),
               (width - marker_offset, height - marker_offset)],
              fill='#3b82f6', width=2)
    draw.line([(width - marker_offset, height - marker_offset - marker_size),
               (width - marker_offset, height - marker_offset)],
              fill='#3b82f6', width=2)

    # Add circular target rings
    for radius in [80, 120, 160]:
        bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
        draw.ellipse(bbox, outline='#475569', width=1)

    # Add text
    try:
        # Try to use a default font
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        # Fallback to default font
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Main text
    view_text = f"VIEW {view_num}"
    placeholder_text = "Awaiting Reconstruction"

    # Calculate text positions
    view_bbox = draw.textbbox((0, 0), view_text, font=font_large)
    view_width = view_bbox[2] - view_bbox[0]
    view_height = view_bbox[3] - view_bbox[1]

    placeholder_bbox = draw.textbbox((0, 0), placeholder_text, font=font_small)
    placeholder_width = placeholder_bbox[2] - placeholder_bbox[0]

    # Draw text with shadow effect
    shadow_offset = 2
    draw.text((center_x - view_width // 2 + shadow_offset, center_y - 40 + shadow_offset),
              view_text, fill='#000000', font=font_large)
    draw.text((center_x - view_width // 2, center_y - 40),
              view_text, fill='#3b82f6', font=font_large)

    draw.text((center_x - placeholder_width // 2 + shadow_offset, center_y + 20 + shadow_offset),
              placeholder_text, fill='#000000', font=font_small)
    draw.text((center_x - placeholder_width // 2, center_y + 20),
              placeholder_text, fill='#64748b', font=font_small)

    # Add metadata labels at bottom
    metadata_text = f"Ready for k-space data • Resolution: {width}×{height}"
    metadata_bbox = draw.textbbox((0, 0), metadata_text, font=font_small)
    metadata_width = metadata_bbox[2] - metadata_bbox[0]
    draw.text((center_x - metadata_width // 2, height - 40),
              metadata_text, fill='#475569', font=font_small)

    return img


# Compact Header
st.markdown('<h1 class="main-title">🏥 AI-Powered MRI Reconstruction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Medical Image Processing & Visualization</p>', unsafe_allow_html=True)

# Compact Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    uploaded_file = st.file_uploader("📁 Upload K-Space", type=["mat"], help="Upload .mat k-space data")
    task = st.selectbox("🎯 Task", ["Cine", "Mapping"], help="Reconstruction task type")

    with st.expander("🔧 Advanced", expanded=False):
        batch_size = st.slider("Batch", 1, 8, 4, help="Batch size")
        num_workers = st.slider("Workers", 1, 8, 2, help="Worker threads")
        num_cascades = st.slider("Cascades", 8, 24, 16, help="Model cascades")
        center_crop = st.checkbox("Center Crop", value=True, help="Apply center crop")

    if uploaded_file:
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.success(f"✓ {uploaded_file.name}")
        st.metric("Size", f"{file_size:.1f} MB")

        st.markdown("---")
        # Process button in sidebar
        process_btn = st.button("🚀 Start Reconstruction", use_container_width=True, key="process_btn")
    else:
        st.info("Ready for upload")

        st.markdown("---")
        # Disabled button when no file
        st.button("🚀 Start Reconstruction", use_container_width=True, disabled=True,
                  help="Please upload a k-space file first")

    # ---- NEW: Dynamic display controls ----
    st.markdown("---")
    st.markdown("### 🖼 Display Controls (demo mode)")
    # Universal size
    img_width = st.slider("Image width (px)", 80, 800, 250, step=10)
    img_height = st.slider("Image height (px)", 80, 800, 250, step=10)
    st.markdown("**Per-image offsets (px)**")
    # Per-image top and left offsets
    top_offsets = [
        st.slider(f"Top offset V1", -200, 200, 0, key="top1"),
        st.slider(f"Top offset V2", -200, 200, 0, key="top2"),
        st.slider(f"Top offset V3", -200, 200, 0, key="top3"),
        st.slider(f"Top offset V4", -200, 200, 0, key="top4"),
    ]
    left_offsets = [
        st.slider(f"Left offset V1", -200, 200, 0, key="left1"),
        st.slider(f"Left offset V2", -200, 200, 0, key="left2"),
        st.slider(f"Left offset V3", -200, 200, 0, key="left3"),
        st.slider(f"Left offset V4", -200, 200, 0, key="left4"),
    ]

# Initialize session state (unchanged)
if 'reconstructed_images' not in st.session_state:
    st.session_state.reconstructed_images = []

# Initialize placeholder images if no reconstruction has been done
if 'placeholder_images' not in st.session_state:
    st.session_state.placeholder_images = [
        {
            'image': create_placeholder_image(i + 1),
            'label': f'View {i + 1}',
            'index': i
        }
        for i in range(4)
    ]

# Main processing logic (unchanged demo-loading section)
if uploaded_file is not None:
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        script_dir = os.getcwd()

    repo_dir = os.path.abspath(os.path.join(script_dir, "PromptMR"))

    if not os.path.isdir(repo_dir):
        st.error(f"❌ PromptMR directory not found at: {repo_dir}")
        st.stop()

    # Check if process button was clicked
    if process_btn:
        with st.spinner('Loading pre-reconstructed demo images...'):
            # Path where your demo reconstructed images are stored
            demo_image_dir = os.path.join(script_dir, "demo_images")

            # Expected filenames (you can change these)
            demo_files = [
                "kspace.png",
                "grappa_recon.png",
                "sense_recon.png",
                "model_recon.png"
            ]

            output_images = []
            for i, fname in enumerate(demo_files):
                img_path = os.path.join(demo_image_dir, fname)
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    if i == 0:
                        label = "Input K-Space"
                    elif i == 1:
                        label = "GRAPPA Reconstruction"
                    elif i == 2:
                        label = "SENSE Reconstruction"
                    else:
                        label = "AI Model Reconstruction"

                    output_images.append({
                        'image': img,
                        'label': label,
                        'index': i
                    })
                else:
                    st.warning(f"⚠️ Missing demo image: {fname}")

            if output_images:
                st.session_state.reconstructed_images = output_images
                st.success("✅ Demo reconstruction displayed successfully!")
                st.info("🧠 Demo Mode: Showing pre-computed reconstruction results for visualization purposes only.")
            else:
                st.error("❌ No demo images found in demo_images folder.")

# Determine which images to display (reconstructed or placeholder)
if st.session_state.reconstructed_images:
    images = st.session_state.reconstructed_images
else:
    images = st.session_state.placeholder_images


# ---------------------------
# Helper: convert PIL image to base64 for inline HTML with styles
# ---------------------------
def pil_to_base64_html(img_pil, width_px=None, height_px=None, style=""):
    buf = BytesIO()
    img = img_pil.copy()
    # resize to requested pixel dims for faster browser rendering (preserve aspect)
    if width_px and height_px:
        img.thumbnail((width_px, height_px), Image.LANCZOS)
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    style_attr = f'style="{style}"' if style else ""
    return f'<img src="data:image/png;base64,{b64}" {style_attr}/>'


# ---------------------------
# Display the 4 images with per-image offsets and size from sidebar
# ---------------------------
# Build inline HTML where each image is wrapped in a div with margin offsets
img_html_parts = []
for idx, img_obj in enumerate(images[:4]):
    top = top_offsets[idx] if idx < len(top_offsets) else 0
    left = left_offsets[idx] if idx < len(left_offsets) else 0
    # Inline CSS: margin-top and margin-left in pixels, display inline-block for side-by-side
    style = f"display:inline-block; margin-top:{top}px; margin-left:{left}px; margin-right:20px;"
    # convert to base64 img tag (thumbnail to requested dimensions)
    img_tag = pil_to_base64_html(img_obj['image'], width_px=img_width, height_px=img_height, style=style)
    label_html = f'<div style="text-align:center;font-weight:600;color:#334155;margin-bottom:6px;">🔍 {img_obj["label"]}</div>'
    img_html_parts.append(f'<div style="display:inline-block; vertical-align:top;">{label_html}{img_tag}</div>')

# Render the combined HTML (wrap in a centered container)
combined_html = '<div style="text-align:center;">' + ''.join(img_html_parts) + '</div>'
st.markdown(combined_html, unsafe_allow_html=True)

# Compact download row (only show if reconstructed images exist)
if st.session_state.reconstructed_images:
    col1, col2, col3, col4 = st.columns(4, gap="small")

    for idx, (col, img_data) in enumerate(zip([col1, col2, col3, col4], images[:4])):
        if idx < len(images):
            with col:
                buf = BytesIO()
                img_data['image'].save(buf, format='PNG')
                st.download_button(
                    label=f"⬇️ V{idx + 1}",
                    data=buf.getvalue(),
                    file_name=f"view_{idx + 1}.png",
                    mime="image/png",
                    use_container_width=True
                )
