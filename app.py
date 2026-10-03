"""
Image Recognition App
Classifies an image into 1,000 ImageNet categories using a pretrained
Vision Transformer / ResNet model from Hugging Face. Runs fully offline
after the first model download. No API key needed.
"""
import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Image Recognition", page_icon="🖼️", layout="centered")

# ---------------- Custom CSS (dark, 3D) ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Sora:wght@600;700&display=swap');

    :root {
        --bg: #0C0F1D;
        --bg-2: #121629;
        --raise-top: #262D4D;
        --raise-bottom: #161A30;
        --edge-light: rgba(255, 255, 255, 0.09);
        --edge-dark: rgba(0, 0, 0, 0.55);
        --ink: #E8EBFA;
        --muted: #9AA3C7;
        --accent: #7C6CFF;
        --accent-2: #4F8CFF;
        --accent-deep: #3A2FB0;
        --good: #3DDC97;
        --radius: 14px;
    }

    /* ---------- Page ---------- */
    .stApp {
        background:
            radial-gradient(1200px 600px at 50% -10%, #1C2247 0%, transparent 60%),
            linear-gradient(180deg, var(--bg-2) 0%, var(--bg) 100%);
        background-attachment: fixed;
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    .block-container {
        max-width: 780px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ---------- Typography ---------- */
    h1, h2, h3 {
        font-family: 'Sora', sans-serif;
        color: var(--ink);
        letter-spacing: -0.02em;
    }
    h1 {
        font-size: 2.5rem;
        line-height: 1.15;
        text-shadow: 0 3px 0 rgba(0, 0, 0, 0.55), 0 8px 18px rgba(124, 108, 255, 0.35);
    }
    p, label, .stMarkdown, [data-testid="stWidgetLabel"] p {
        color: var(--ink);
        line-height: 1.6;
    }
    [data-testid="stCaptionContainer"], [data-testid="stImageCaption"] {
        color: var(--muted);
    }
    hr {
        border-color: var(--edge-light);
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #151A33 0%, #0E1224 100%);
        border-right: 1px solid var(--edge-light);
        box-shadow: 8px 0 30px rgba(0, 0, 0, 0.45);
    }
    [data-testid="stSidebar"] h2 {
        font-size: 1.1rem;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: none;
        padding: 0.4rem;
        background: #0A0D1A;
        border-radius: var(--radius);
        box-shadow: inset 0 3px 8px rgba(0, 0, 0, 0.7), 0 1px 0 var(--edge-light);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.55rem 1.2rem;
        color: var(--muted);
        font-weight: 500;
        border-radius: 10px;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #FFFFFF;
        background: linear-gradient(180deg, var(--raise-top), var(--raise-bottom));
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.55), inset 0 1px 0 var(--edge-light);
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* ---------- File uploader and camera: raised cards ---------- */
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(160deg, var(--raise-top) 0%, var(--raise-bottom) 100%);
        border: 1px solid var(--edge-light);
        border-radius: var(--radius);
        padding: 1.9rem;
        box-shadow:
            0 14px 28px rgba(0, 0, 0, 0.55),
            0 2px 0 rgba(0, 0, 0, 0.6),
            inset 0 1px 0 var(--edge-light);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        transform: translateY(-3px);
        box-shadow:
            0 20px 36px rgba(0, 0, 0, 0.6),
            0 2px 0 rgba(0, 0, 0, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.14);
    }
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] span {
        color: var(--muted);
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(180deg, #343C66, #232948);
        color: var(--ink);
        border: 1px solid var(--edge-light);
        border-radius: 10px;
        box-shadow: 0 4px 0 #0B0E1C, 0 8px 14px rgba(0, 0, 0, 0.5);
    }
    [data-testid="stFileUploaderDropzone"] button:active {
        transform: translateY(3px);
        box-shadow: 0 1px 0 #0B0E1C, 0 3px 6px rgba(0, 0, 0, 0.5);
    }
    [data-testid="stCameraInput"] {
        background: linear-gradient(160deg, var(--raise-top) 0%, var(--raise-bottom) 100%);
        border: 1px solid var(--edge-light);
        border-radius: var(--radius);
        padding: 0.9rem;
        box-shadow: 0 14px 28px rgba(0, 0, 0, 0.55), inset 0 1px 0 var(--edge-light);
    }

    /* ---------- Images: framed and lifted ---------- */
    [data-testid="stImage"] img {
        border-radius: var(--radius);
        border: 1px solid var(--edge-light);
        box-shadow:
            0 18px 40px rgba(0, 0, 0, 0.65),
            0 4px 0 rgba(0, 0, 0, 0.6),
            0 0 0 6px #151A30;
        transition: transform 0.25s ease;
    }
    [data-testid="stImage"] img:hover {
        transform: perspective(900px) rotateX(2deg) translateY(-4px);
    }

    /* ---------- Buttons: pressable 3D keys ---------- */
    .stButton > button {
        background: linear-gradient(180deg, #343C66, #232948);
        color: var(--ink);
        border: 1px solid var(--edge-light);
        border-radius: 12px;
        padding: 0.65rem 1.6rem;
        font-weight: 500;
        box-shadow: 0 5px 0 #0B0E1C, 0 12px 20px rgba(0, 0, 0, 0.5);
        transition: transform 0.08s ease, box-shadow 0.08s ease, filter 0.15s ease;
    }
    .stButton > button:hover {
        filter: brightness(1.12);
        color: #FFFFFF;
        border-color: var(--edge-light);
    }
    .stButton > button:active {
        transform: translateY(4px);
        box-shadow: 0 1px 0 #0B0E1C, 0 4px 8px rgba(0, 0, 0, 0.5);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(180deg, #9488FF 0%, var(--accent) 45%, #5B49E6 100%);
        color: #FFFFFF;
        border: none;
        font-weight: 700;
        box-shadow:
            0 5px 0 var(--accent-deep),
            0 14px 26px rgba(124, 108, 255, 0.45),
            inset 0 1px 0 rgba(255, 255, 255, 0.35);
    }
    .stButton > button[kind="primary"]:active {
        box-shadow:
            0 1px 0 var(--accent-deep),
            0 5px 10px rgba(124, 108, 255, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.25);
    }
    .stButton > button:focus-visible {
        outline: 3px solid rgba(124, 108, 255, 0.55);
        outline-offset: 3px;
    }

    /* ---------- Result and info boxes ---------- */
    [data-testid="stAlert"] {
        background: linear-gradient(160deg, var(--raise-top) 0%, var(--raise-bottom) 100%);
        border: 1px solid var(--edge-light);
        border-radius: var(--radius);
        color: var(--ink);
        box-shadow: 0 12px 26px rgba(0, 0, 0, 0.5), inset 0 1px 0 var(--edge-light);
    }
    [data-testid="stAlert"] p {
        color: var(--ink);
    }

    /* ---------- Progress bars: inset track, glossy fill ---------- */
    .stProgress > div > div > div {
        background-color: #080B16;
        border-radius: 999px;
        height: 0.65rem;
        box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.85), 0 1px 0 var(--edge-light);
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(180deg, #A79DFF 0%, var(--accent) 50%, #4F8CFF 100%);
        border-radius: 999px;
        box-shadow: 0 0 12px rgba(124, 108, 255, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.4);
    }

    /* ---------- Select box and slider: inset controls ---------- */
    [data-baseweb="select"] > div {
        background: #0A0D1A;
        border: 1px solid var(--edge-light);
        border-radius: 12px;
        color: var(--ink);
        box-shadow: inset 0 3px 8px rgba(0, 0, 0, 0.7);
    }
    [data-baseweb="select"] * {
        color: var(--ink);
    }
    [data-baseweb="popover"] ul,
    [data-baseweb="popover"] li {
        background: #151A30;
        color: var(--ink);
    }
    [data-baseweb="popover"] li:hover {
        background: #232948;
    }
    [data-testid="stSlider"] [role="slider"] {
        background: radial-gradient(circle at 35% 30%, #B3AAFF, var(--accent) 60%, #4B3BD0);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.4);
        border: none;
    }
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"],
    [data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
        color: var(--muted);
    }

    /* ---------- Spinner ---------- */
    [data-testid="stSpinner"] p {
        color: var(--muted);
    }

    /* ---------- Mobile ---------- */
    @media (max-width: 640px) {
        .block-container {
            padding-top: 1.5rem;
        }
        h1 {
            font-size: 1.8rem;
        }
    }

    /* ---------- Reduced motion ---------- */
    @media (prefers-reduced-motion: reduce) {
        * {
            transition: none !important;
        }
        [data-testid="stImage"] img:hover,
        [data-testid="stFileUploaderDropzone"]:hover {
            transform: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODELS = {
    "ViT Base (Google)": "google/vit-base-patch16-224",
    "ResNet-50 (Microsoft)": "microsoft/resnet-50",
}


@st.cache_resource(show_spinner="Loading model (first run downloads it, please wait)...")
def load_classifier(model_id: str):
    return pipeline("image-classification", model=model_id)


# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
top_k = st.sidebar.slider("Number of predictions", 1, 10, 5)
st.sidebar.markdown("---")
st.sidebar.caption("Models are trained on ImageNet (1,000 everyday object classes).")

# ---------------- Main ----------------
st.title("🖼️ Image Recognition")
st.write("Upload an image or take a photo, and the model will tell you what it sees.")

tab_upload, tab_camera = st.tabs(["📁 Upload", "📷 Camera"])
image = None

with tab_upload:
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if file:
        image = Image.open(file).convert("RGB")

with tab_camera:
    photo = st.camera_input("Take a picture")
    if photo:
        image = Image.open(photo).convert("RGB")

if image is not None:
    st.image(image, caption="Input image")

    if st.button("🔍 Recognize", type="primary"):
        classifier = load_classifier(MODELS[model_name])
        with st.spinner("Analyzing image..."):
            results = classifier(image, top_k=top_k)

        best = results[0]
        st.success(f"**Top prediction:** {best['label']}  ({best['score'] * 100:.2f}%)")

        st.subheader("All predictions")
        for r in results:
            st.write(f"{r['label']} — {r['score'] * 100:.2f}%")
            st.progress(float(r["score"]))
else:
    st.info("👆 Upload an image or use the camera to get started.")