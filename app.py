import streamlit as st
import tensorflow as tf
import keras # Changed to direct import for compatibility
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import numpy as np
import matplotlib.cm as cm
import warnings
import os

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Forager Vision",
    page_icon="🍄",
    layout="centered",
    initial_sidebar_state="expanded"
)
warnings.filterwarnings("ignore")

# --- 2. PERMANENT DARK THEME ---
# Hardcoded Dark Mode Palette
main_bg = "#0E1117"
text_color = "#FAFAFA"
card_bg = "#262730"
card_border = "#41444C"
safe_bg = "#0F3818"
safe_text = "#7DDA93"
danger_bg = "#3D0F12"
danger_text = "#FF8A8A"
warning_bg = "#332b00"
warning_text = "#ffeeba"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {main_bg}; color: {text_color}; }}
    [data-testid="stSidebar"] {{ background-color: {card_bg}; }}
    .main-header {{ font-size: 2.5rem; font-weight: 700; color: {text_color}; text-align: center; margin-bottom: 20px; }}
    .result-card {{ padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px; background-color: {card_bg}; border: 1px solid {card_border}; }}
    .ux-message {{ text-align: center; font-size: 1.1rem; color: {text_color}; margin-bottom: 20px; }}
    p, h1, h2, h3, label, .stMarkdown, .stText {{ color: {text_color} !important; }}
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD MODELS ---
@st.cache_resource
def load_models():
    # Main Model
    if os.path.exists('mushroom_model_lite.keras'):
        main_model = keras.models.load_model('mushroom_model_lite.keras')
    elif os.path.exists('best_mushroom_model.keras'):
        main_model = keras.models.load_model('best_mushroom_model.keras')
    else:
        return None, None
    
    # Gatekeeper (Mushroom Detector)
    gatekeeper = MobileNetV2(weights='imagenet')
    return main_model, gatekeeper

# --- 4. HELPER FUNCTIONS ---
def is_mushroom(img, gatekeeper_model):
    img_resized = img.resize((224, 224))
    img_array = keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = mobilenet_preprocess(img_array)
    
    preds = gatekeeper_model.predict(img_array)
    decoded = decode_predictions(preds, top=5)[0]
    
    keywords = ['mushroom', 'fungus', 'agaric', 'gyromitra', 'bolete', 'stinkhorn', 'earthstar', 'hen-of-the-woods']
    for _, label, _ in decoded:
        for k in keywords:
            if k in label.lower(): return True
    return False

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    original_activation = model.layers[-1].activation
    model.layers[-1].activation = tf.keras.activations.linear
    
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[0][0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    model.layers[-1].activation = original_activation

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

# --- 5. SIDEBAR ---
with st.sidebar:
    if os.path.exists("logo.jpg"): st.image("logo.jpg", use_container_width=True)
    elif os.path.exists("logo.png"): st.image("logo.png", use_container_width=True)
    else: st.header("🍄")
    
    st.header("Forager Vision")
    st.markdown("---")
    st.info("**System Status:** Online\n\nModel: DenseNet121\nChecks: Toxicity & Species Verification")

# --- 6. MAIN INTERFACE ---
st.markdown('<div class="main-header">Mushroom Safety Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="ux-message">Upload a clear photo of a mushroom to detect potential toxicity.</div>', unsafe_allow_html=True)

try:
    main_model, gatekeeper = load_models()
    if main_model is None:
        st.error("🚨 Model file not found. Please ensure .keras file is in the directory.")
        st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="large")
    
    # FIX: Convert to RGB immediately to handle 4-channel PNGs
    image_pil = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("📸 Specimen")
        st.image(image_pil, use_container_width=True, caption="Original Image")

    with col2:
        st.subheader("🧬 Analysis")
        with st.spinner('Scanning bio-markers...'):
            
            # STEP 1: IMPROVED GATEKEEPER CHECK
            if not is_mushroom(image_pil, gatekeeper):
                st.markdown(f"""
                    <div class="result-card" style="background-color: {warning_bg}; border: 1px solid {warning_text};">
                        <h3 style="color: {warning_text}; margin:0;">🤔 Object Not Recognized</h3>
                        <p style="color: {warning_text}; margin-top: 10px;">
                            We couldn't detect a mushroom in this image.
                            <br><br>
                            To ensure accurate safety analysis, please <b>upload a clear, centered photo</b> where the mushroom is the main subject.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.stop() # Stops execution here. No prediction happens.

            # STEP 2: PREDICT TOXICITY
            img_resized = ImageOps.fit(image_pil, (224, 224), Image.Resampling.LANCZOS)
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = densenet_preprocess(img_array)

            prediction = main_model.predict(img_array)
            score = prediction[0][0]
            
            if score > 0.5:
                label = "POISONOUS"
                confidence = score * 100
                dyn_bg = danger_bg
                dyn_text = danger_text
                icon = "☣️"
            else:
                label = "EDIBLE"
                confidence = (1 - score) * 100
                dyn_bg = safe_bg
                dyn_text = safe_text
                icon = "🥗"

            st.markdown(f"""
                <div class="result-card" style="background-color: {dyn_bg}; border: 2px solid {dyn_text};">
                    <h1 style="margin:0; color: {dyn_text};">{icon} {label}</h1>
                    <h3 style="margin:10px 0 0 0; color: {dyn_text};">Confidence: {confidence:.2f}%</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.progress(int(confidence))

    # --- 7. XAI SECTION ---
    st.divider()
    st.subheader("🔬 Visual Evidence (Grad-CAM)")
    
    try:
        last_conv_layer_name = ""
        for layer in main_model.layers[::-1]:
            if "conv" in layer.name:
                last_conv_layer_name = layer.name
                break
        
        heatmap = make_gradcam_heatmap(img_array, main_model, last_conv_layer_name)
        
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        
        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img_resized.width, img_resized.height))
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)
        
        superimposed_img = jet_heatmap * 0.4 + keras.utils.img_to_array(img_resized)
        superimposed_img = keras.utils.array_to_img(superimposed_img)
        
        xc1, xc2 = st.columns(2)
        with xc1: st.image(img_resized, caption="Original", use_container_width=True)
        with xc2: st.image(superimposed_img, caption="AI Attention Heatmap", use_container_width=True)
            
    except:
        st.warning("Heatmap unavailable.")