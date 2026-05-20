from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
import pickle
import os

def create_cnn_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def create_vgg16_model():
    conv_base = VGG16(weights='imagenet', 
                      include_top=False, 
                      input_shape=(32, 32, 3))
    conv_base.trainable = False
    
    vgg_model = keras.Sequential([
        conv_base,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    vgg_model.compile(optimizer=keras.optimizers.Adam(0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    return vgg_model

def load_models():
    models_exist = all([
        os.path.exists("fashion_cnn.h5"),
        os.path.exists("fashion_vgg16.h5"),
        os.path.exists("history_cnn.pkl"),
        os.path.exists("history_vgg.pkl")
    ])
    
    if not models_exist:
        raise FileNotFoundError(
            "Models not found. Please run train.py first to train and save models."
        )
    
    cnn_model = keras.models.load_model("fashion_cnn.h5")
    vgg_model = keras.models.load_model("fashion_vgg16.h5")
    
    with open('history_cnn.pkl', 'rb') as f:
        history_cnn = pickle.load(f)
    with open('history_vgg.pkl', 'rb') as f:
        history_vgg = pickle.load(f)
    
    return cnn_model, vgg_model, history_cnn, history_vgg