import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageOps
import pickle 


st.set_page_config(page_title="Fashion MNIST Classifier", layout="wide")
st.title("Fashion MNIST Classifier")
st.write("Upload an image of clothing and I will predict what type of clothing it is!")


CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def create_cnn_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(63, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

#CNN
@st.cache_resource
def train_models():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_train_cnn = x_train.reshape(-1, 28, 28, 1)
    cnn_model = create_cnn_model()
    history_cnn = cnn_model.fit(x_train_cnn, y_train, epochs=10, batch_size=128, validation_split=0.2, verbose=1) 

    #VGG16
    import tensorflow as tf
    from tensorflow.keras.applications import VGG16

    x_train_vgg = tf.image.resize(x_train[..., tf.newaxis], [32, 32]).numpy()
    x_train_vgg = np.repeat(x_train_vgg, 3, axis=-1)

    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    conv_base.trainable = False

    vgg_model = keras.Sequential([
    conv_base,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
    ])
    vgg_model.compile(optimizer=keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history_vgg = vgg_model.fit(x_train_vgg, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=1)

    return cnn_model, vgg_model, history_cnn.history, history_vgg.history 

try:
    cnn_model = keras.models.load_model("fashion_cnn.h5")
    vgg_model = keras.models.load_model("fashion_vgg16.h5")
    with open('history_cnn.pkl', 'rb') as f:
        history_cnn = pickle.load(f)
    with open('history_vgg.pkl', 'rb') as f:
        history_vgg = pickle.load(f)
    st.sidebar.write("Models loaded successfully!")
except:
    st.sidebar.warning("Model training (will take 5 minutes)")
    cnn_model, vgg_model, history_cnn, history_vgg = train_models()
    cnn_model.save("fashion_cnn.h5")
    vgg_model.save("fashion_vgg16.h5")
    with open('history_cnn.pkl', 'wb') as f:
        pickle.dump(history_cnn, f)
    with open('history_vgg.pkl', 'wb') as f:
        pickle.dump(history_vgg, f)
    st.sidebar.write("Models trained and saved!")

st.sidebar.header("Settings")
mdoel_choice = st.sidebar.radio("Choose Model", ["CNN", "VGG16"])

st.sidebar.header("Training schedules")
if mdoel_choice == "CNN":
    hist = history_cnn
    input_size = 28
    model = cnn_model
else:
    hist = history_vgg
    input_size = 32
    model = vgg_model 

fig_acc, ax_acc = plt.subplots(figsize=(5, 3))
ax_acc.plot(hist['accuracy'], 'b-', label='Train')
ax_acc.plot(hist['val_accuracy'], 'r-', label='Validaiton')
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
    
    image_resized = image.resize((input_size, input_size))
    img_array = np.array(image_resized).astype('float32') / 255.0

    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array
    if model_choice == "CNN":
        img_array = img_array.reshape(1, 28, 28, 1)
    else:
        img_array = img_array.reshape(1, 32, 32, 1)
        img_array = np.repeat(img_array, 3, axis=-1)

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class] * 100

    with col2:
        st.subheader('Result')
        if confidence > 80:
            st.success(f"Predicted class: {CLASS_NAMES[predicted_class]}")
        elif confidence > 50:
            st.warning(f"Predicted class: {CLASS_NAMES[predicted_class]}")
        else:
            st.error(f"Predicted class: {CLASS_NAMES[predicted_class]}")
        st.metric("Confidence", f"{confidence:.2f}%")

    st.subheader('Probabilities by class')
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2ecc71' if i == predicted_class else '#3498db' for i in range(10)]
    bars = ax.barh(CLASS_NAMES, predictions * 100, color=colors)
    ax.set_xlabel('Probability(%)')
    ax.set_xlim(0, 100)
    for bar, prop in zip(bars, predictions * 100):
        if prop > 3:
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{prop:.1f}%', 
                    va='center')
    st.pyplot(fig)
    st.subheader("Probability table")
    import pandas as pd
    prob_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability":[f'{p*100:.2f}%' for p in predictions]})
    def highlight(row):
        if row['Class'] == CLASS_NAMES[predicted_class]:
            return ['background-color: #2ecc71; color: white'] * len(row)
        return [''] * len(row)

    st.dataframe(prob_df.style.apply(highlight, axis=1), use_container_width=True)



