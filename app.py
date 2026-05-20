import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from model import load_models
from utils import preprocess_image, get_input_size

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

st.set_page_config(page_title="Fashion MNIST Classifier", layout="wide")
st.title("Fashion MNIST Classifier")
st.write("Upload an image of clothing and I will predict what type of clothing it is!")

@st.cache_resource
def load_cached_models():
    try:
        cnn_model, vgg_model, history_cnn, history_vgg = load_models()
        st.sidebar.success("✅ Models loaded successfully!")
        return cnn_model, vgg_model, history_cnn, history_vgg
    except FileNotFoundError as e:
        st.error(f"❌ Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error loading models: {e}")
        st.stop()

cnn_model, vgg_model, history_cnn, history_vgg = load_cached_models()

st.sidebar.header("Settings")
model_choice = st.sidebar.radio(
    "Choose Model", 
    ["CNN", "VGG16"],
    key="model_selector"
)

if model_choice == "CNN":
    hist = history_cnn
    model = cnn_model
else:
    hist = history_vgg
    model = vgg_model

input_size = get_input_size(model_choice)

st.sidebar.header("Training History")

fig_acc, ax_acc = plt.subplots(figsize=(5, 3))
ax_acc.plot(hist['accuracy'], 'b-', label='Train')
ax_acc.plot(hist['val_accuracy'], 'r-', label='Validation')  # ИСПРАВЛЕНО: опечатка
ax_acc.set_title('Accuracy')
ax_acc.legend()
ax_acc.grid(True)
st.sidebar.pyplot(fig_acc)

fig_loss, ax_loss = plt.subplots(figsize=(5, 3))
ax_loss.plot(hist['loss'], 'b-', label='Train')
ax_loss.plot(hist['val_loss'], 'r-', label='Validation')
ax_loss.set_title('Loss')
ax_loss.legend()
ax_loss.grid(True)
st.sidebar.pyplot(fig_loss)

st.header("Upload an image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file).convert('L')
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    img_array = preprocess_image(image, model_choice, input_size)
    
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class] * 100
    
    with col2:
        st.subheader('Result')
        if confidence > 80:
            st.success(f"Predicted class: **{CLASS_NAMES[predicted_class]}**")
        elif confidence > 50:
            st.warning(f"Predicted class: **{CLASS_NAMES[predicted_class]}**")
        else:
            st.error(f"Predicted class: **{CLASS_NAMES[predicted_class]}**")
        st.metric("Confidence", f"{confidence:.2f}%")
    
    st.subheader('Probabilities by class')
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2ecc71' if i == predicted_class else '#3498db' for i in range(10)]
    bars = ax.barh(CLASS_NAMES, predictions * 100, color=colors)
    ax.set_xlabel('Probability (%)')
    ax.set_xlim(0, 100)
    
    for bar, prob in zip(bars, predictions * 100):
        if prob > 3:
            ax.text(bar.get_width() + 1, 
                   bar.get_y() + bar.get_height()/2, 
                   f'{prob:.1f}%', 
                   va='center')
    
    st.pyplot(fig)
    
    st.subheader("Probability Table")
    prob_df = pd.DataFrame({
        "Class": CLASS_NAMES, 
        "Probability": [f'{p*100:.2f}%' for p in predictions]
    })
    
    def highlight_predicted(row):
        if row['Class'] == CLASS_NAMES[predicted_class]:
            return ['background-color: #2ecc71; color: white'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        prob_df.style.apply(highlight_predicted, axis=1), 
        use_container_width=True
    )