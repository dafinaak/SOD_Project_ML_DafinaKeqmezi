from pathlib import Path
import io
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import streamlit as st

from sod_model import SODNet

CKPT_PATH   = Path(r"C:\Users\dafin\ml_checkpoints\baseline_best.pt")
LEGACY_PATH = Path(r"C:\Users\dafin\ml_checkpoints\best_model.pt")
IMAGE_SIZE  = 128
THRESHOLD   = 0.5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    """Load the model once and cache it across user interactions."""
    ckpt = CKPT_PATH if CKPT_PATH.exists() else LEGACY_PATH
    model = SODNet(depth=4).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model, ckpt


def preprocess(pil_img):
    """Resize and normalize an uploaded PIL image into a model-ready tensor."""
    img = pil_img.convert("RGB")
    img = TF.resize(img, [IMAGE_SIZE, IMAGE_SIZE], interpolation=TF.InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(img).unsqueeze(0)  # (1, 3, H, W) in [0, 1]
    return tensor, np.array(img) / 255.0


def make_overlay(img_np, mask_np, color=(1.0, 0.0, 0.0), alpha=0.5):
    """Blend a colored mask on top of the image where the mask is positive."""
    overlay = img_np.copy()
    layer = np.zeros_like(overlay)
    for c in range(3):
        layer[..., c] = color[c]
    m3 = np.repeat(mask_np[..., None], 3, axis=2)
    return np.clip(np.where(m3 > 0.5, (1 - alpha) * overlay + alpha * layer, overlay), 0, 1)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Salient Object Detection Demo", layout="wide")
st.title("Salient Object Detection — Demo")
st.caption("Upload an image. The model will identify the most visually important region.")

model, ckpt_used = load_model()
st.caption(f"Using checkpoint: `{ckpt_used.name}` on device `{DEVICE}`")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil_img = Image.open(io.BytesIO(uploaded.read()))
    tensor, img_np = preprocess(pil_img)

    with torch.no_grad():
        t0 = time.time()
        prob = model(tensor.to(DEVICE))[0, 0].cpu().numpy()
        inference_ms = (time.time() - t0) * 1000

    pred_mask = (prob > THRESHOLD).astype(np.float32)
    overlay = make_overlay(img_np, pred_mask)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Input (resized)**")
        st.image(img_np, use_container_width=True)
    with col2:
        st.markdown("**Probability map**")
        st.image(prob, use_container_width=True, clamp=True)
    with col3:
        st.markdown("**Predicted mask**")
        st.image(pred_mask, use_container_width=True, clamp=True)
    with col4:
        st.markdown("**Overlay**")
        st.image(overlay, use_container_width=True, clamp=True)

    st.markdown("---")
    st.markdown("**Statistics**")
    st.write(f"Inference time: {inference_ms:.1f} ms")
    st.write(f"Salient pixels: {int(pred_mask.sum())} / {pred_mask.size} "
             f"({100 * pred_mask.mean():.1f}%)")
    st.write(f"Max probability: {prob.max():.3f}")
    st.write(f"Mean probability: {prob.mean():.3f}")
else:
    st.info("← Upload an image to see results.")