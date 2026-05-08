from pathlib import Path
import io
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import streamlit as st

from sod_model import SODNet

CKPT_DIR    = Path(r"C:\Users\dafin\ml_checkpoints")
IMAGE_SIZE  = 128
THRESHOLD   = 0.5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "Baseline (depth=4)":          ("baseline_best.pt",             4),
    "Exp 1 — Strong augmentation": ("exp1_strong_aug_best.pt",      4),
    "Exp 2 — Deeper + lower LR":   ("exp2_deeper_lower_lr_best.pt", 5),
}


@st.cache_resource
def load_model(ckpt_filename: str, depth: int):
    """Load a model from a checkpoint and cache it per checkpoint."""
    ckpt = CKPT_DIR / ckpt_filename
    model = SODNet(depth=depth).to(DEVICE)
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


st.set_page_config(page_title="Salient Object Detection Demo", layout="wide")
st.title("Salient Object Detection — Demo")
st.caption("Upload an image. The model will identify the most visually important region.")

selected_label = st.selectbox(
    "Choose a model",
    list(MODELS.keys()),
    index=0,
    help="Switch between the baseline and your experiments to compare results on the same image.",
)
ckpt_filename, depth = MODELS[selected_label]

try:
    model, ckpt_used = load_model(ckpt_filename, depth)
except FileNotFoundError:
    st.error(f"Checkpoint not found: {CKPT_DIR / ckpt_filename}")
    st.stop()

st.caption(f"Using checkpoint: `{ckpt_used.name}` (depth={depth}) on device `{DEVICE}`")

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
    st.write(f"Model: {selected_label}")
    st.write(f"Inference time: {inference_ms:.1f} ms")
    st.write(f"Salient pixels: {int(pred_mask.sum())} / {pred_mask.size} "
             f"({100 * pred_mask.mean():.1f}%)")
    st.write(f"Max probability: {prob.max():.3f}")
    st.write(f"Mean probability: {prob.mean():.3f}")
else:
    st.info("← Upload an image to see results.")