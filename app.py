import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from models.swin_model import create_swin_model

st.set_page_config(
    page_title="Alzheimer's MRI Recognition",
    layout="wide"
)

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "best_swin_tiny_v5.pth"
DEVICE = torch.device("cpu")
CLASSES = ["AD", "NC"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #172554, #312e81);
    color: #f8fafc;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3, p, label, .stMarkdown {
    color: #f8fafc;
}
.header {
    background: linear-gradient(110deg, #6d28d9, #2563eb, #0891b2);
    padding: 32px;
    border-radius: 22px;
    margin-bottom: 28px;
    box-shadow: 0 12px 30px #0004;
}
.header h1 {
    color: white;
    font-size: 34px;
    line-height: 1.3;
}
.header p {
    color: #e0f2fe;
    font-size: 17px;
}
.panel {
    background: #1e2d52;
    border: 1px solid #52658d;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 22px;
}
div[data-testid="stFileUploader"] {
    background: #f8fafc;
    padding: 14px;
    border-radius: 12px;
}
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploader"] span {
    color: #172554 !important;
}
.stButton > button {
    width: 100%;
    min-height: 50px;
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white !important;
    border: none;
    border-radius: 12px;
    font-weight: 700;
}
.stButton > button p {
    color: white !important;
}
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #3730a3, #075985);
    border: 1px solid #60a5fa;
    padding: 20px;
    border-radius: 16px;
}
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricValue"] {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = create_swin_model(num_classes=2)
    state = torch.load(
        MODEL_PATH, map_location=DEVICE, weights_only=True
    )
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


st.markdown("""
<div class="header">
    <h1>An MRI-Based Swin Transformer Approach for Automated Alzheimer's Disease Recognition</h1>
    <p>M.Tech Research Project | Swin Transformer-Tiny</p>
    <p>Binary MRI image classification: AD and NC</p>
</div>
""", unsafe_allow_html=True)

upload_col, info_col = st.columns([1.15, 0.85], gap="large")

with upload_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("MRI Image Upload")
    st.write("Select an MRI image in JPG, JPEG, or PNG format.")

    uploaded = st.file_uploader(
        "Choose MRI image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Selected MRI image", use_container_width=True)
    else:
        st.info("Your selected image will appear here.")
    st.markdown("</div>", unsafe_allow_html=True)

with info_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Model Details")
    st.write("**Architecture:** Swin Transformer-Tiny")
    st.write("**Checkpoint:** Experiment 5")
    st.write("**Input resolution:** 224 × 224")
    st.write("**Output classes:** AD and NC")
    st.markdown("</div>", unsafe_allow_html=True)

    predict = st.button(
        "Run MRI Classification",
        type="primary",
        disabled=uploaded is None
    )

if predict and uploaded is not None:
    if not MODEL_PATH.exists():
        st.error("The trained model checkpoint was not found.")
        st.stop()

    try:
        with st.spinner("Running image classification..."):
            image = Image.open(uploaded).convert("RGB")
            x = transform(image).unsqueeze(0).to(DEVICE)
            model = load_model()

            with torch.inference_mode():
                output = model(x)
                probs = torch.softmax(output, dim=1)[0]

            scores = probs.cpu().tolist()
            index = int(probs.argmax().item())
            prediction = CLASSES[index]

        st.markdown("---")
        st.header("Classification Results")

        a, b, c = st.columns(3)
        a.metric("Predicted Class", prediction)
        b.metric("AD Score", f"{scores[0] * 100:.2f}%")
        c.metric("NC Score", f"{scores[1] * 100:.2f}%")

        st.subheader("Class Score Distribution")
        st.write(f"AD: {scores[0] * 100:.2f}%")
        st.progress(scores[0])
        st.write(f"NC: {scores[1] * 100:.2f}%")
        st.progress(scores[1])

        st.warning(
            "Research prototype only. These scores are not clinical "
            "probabilities and must not be used for medical diagnosis."
        )

    except Exception as error:
        st.error(f"Classification failed: {error}")

st.markdown("---")
st.caption("M.Tech Research Project | Swin Transformer-Tiny | Experiment 5")
