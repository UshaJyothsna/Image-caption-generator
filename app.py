import os
import warnings
import logging
import io
from PIL import Image
import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS

# ---------------- Suppress logs ----------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# ---------- Streamlit setup ----------
st.set_page_config(page_title="Caption Generator App", page_icon="📷")
st.title("📷 Image Caption Generator")
st.markdown(
    "Upload an image, and this app will generate a caption using a "
    "**pretrained BLIP transformer model** and also read it out loud."
)

# ---------- Load model once ----------
@st.cache_resource(show_spinner=True)
def load_blip():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        use_fast=True
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device

processor, model, device = load_blip()

# ---------- Caption generation ----------
@torch.inference_mode()
def generate_caption(pil_image: Image.Image, max_new_tokens: int = 30) -> str:
    image = pil_image.convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.strip()

# ---------- Text-to-Speech ----------
def play_caption(caption: str):
    audio_path = "caption.mp3"
    tts = gTTS(text=caption, lang="en")
    tts.save(audio_path)
    with open(audio_path, "rb") as audio:
        st.audio(audio, format="audio/mp3")

# ---------- UI ----------
uploaded_image = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    pil_img = Image.open(io.BytesIO(uploaded_image.read()))

    st.subheader("Uploaded Image")
    st.image(pil_img, caption="Uploaded Image", width=500)

    st.subheader("Generated Caption")
    with st.spinner("Generating caption..."):
        try:
            caption = generate_caption(pil_img)
            st.success("✅ Caption generated successfully!")
        except Exception as e:
            st.error(f"⚠️ Caption generation failed: {e}")
            caption = None

    if caption:
        st.markdown(
            f"""
            <div style="border-left: 6px solid #ccc;
                        padding: 10px 20px;
                        margin-top: 20px;
                        background:#f9f9f9;">
                <p style="font-style: italic; font-size:18px;">
                    “{caption}”
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        play_caption(caption)
