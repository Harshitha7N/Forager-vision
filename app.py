import streamlit as st
import tensorflow as tf
from tensorflow import keras
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

# Custom CSS for professional styling
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .safe {
        background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb;
    }
    .danger {
        background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb;
    }
    .warning {
        background-color: #fff3cd; color: #856404; border: 2px solid #ffeeba;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODELS (Smart Loader) ---
@st.cache_resource
def load_models():
    # Try loading the Lite model first (for Cloud/GitHub)
    if os.path.exists('mushroom_model_lite.keras'):
        main_model = keras.models.load_model('mushroom_model_lite.keras')
    # Fallback to the big model (for Local)
    elif os.path.exists('best_mushroom_model.keras'):
        main_model = keras.models.load_model('best_mushroom_model.keras')
    else:
        raise FileNotFoundError("No model file found! Please upload 'mushroom_model_lite.keras'.")
    
    # Load Gatekeeper (MobileNetV2) for mushroom detection
    gatekeeper = MobileNetV2(weights='imagenet')
    return main_model, gatekeeper

# --- 3. HELPER FUNCTIONS ---
def is_mushroom(img, gatekeeper_model):
    # Resize for MobileNetV2
    img_resized = img.resize((224, 224))
    img_array = keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = mobilenet_preprocess(img_array)
    
    preds = gatekeeper_model.predict(img_array)
    decoded = decode_predictions(preds, top=5)[0]
    
    # Keywords to check against
    mushroom_keywords = ['mushroom', 'fungus', 'agaric', 'gyromitra', 'bolete', 'stinkhorn', 'earthstar', 'hen-of-the-woods']
    
    for _, label, _ in decoded:
        for keyword in mushroom_keywords:
            if keyword in label.lower():
                return True
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

# --- 4. SIDEBAR ---
with st.sidebar:
    # Logo Logic
    if os.path.exists("logo.jpg"):
        st.image("logo.jpg", use_container_width=True)
    elif os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.header("🍄")
    
    st.title("Forager Vision")
    st.info("AI-Powered Mushroom Safety Tool\n\nDataset: 12,000+ Images\nModel: DenseNet121")

# --- 5. MAIN INTERFACE ---
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Mushroom Safety Analyzer</h1>", unsafe_allow_html=True)
st.write("Upload a photo to detect toxicity.")

try:
    main_model, gatekeeper = load_models()
except Exception as e:
    st.error(f"🚨 Error: {e}")
    st.stop()

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="medium")
    image_pil = Image.open(uploaded_file)
    
    with col1:
        st.subheader("📷 Specimen")
        st.image(image_pil, use_container_width=True, caption="Uploaded Image")

    with col2:
        st.subheader("🧬 Analysis")
        with st.spinner('Scanning...'):
            
            # STEP 1: GATEKEEPER CHECK
            if not is_mushroom(image_pil, gatekeeper):
                st.markdown(f"""
                    <div class="result-card warning">
                        <h2 style="color: #856404; margin:0;">⚠️ Not Recognized</h2>
                        <p>This does not appear to be a mushroom.</p>
                    </div>
                """, unsafe_allow_html=True)
                if not st.checkbox("I confirm this is a mushroom"):
                    st.stop()

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
                css_class = "danger"
                icon = "☣️"
            else:
                label = "EDIBLE"
                confidence = (1 - score) * 100
                css_class = "safe"
                icon = "🥗"

            st.markdown(f"""
                <div class="result-card {css_class}">
                    <h1 style="margin:0;">{icon} {label}</h1>
                    <h3 style="margin:10px 0 0 0;">Confidence: {confidence:.2f}%</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.progress(int(confidence))

    # --- 6. XAI HEATMAP ---
    st.divider()
    with st.expander("🔬 View Explainable AI (Heatmap)"):
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