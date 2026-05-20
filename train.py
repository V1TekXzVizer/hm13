import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.applications import VGG16
import pickle

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0

def create_cnn_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),  # Исправил 63 на 64
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

def train_and_save_models():
    print("Training CNN model...")
    # CNN
    x_train_cnn = x_train.reshape(-1, 28, 28, 1)
    cnn_model = create_cnn_model()
    history_cnn = cnn_model.fit(
        x_train_cnn, y_train, 
        epochs=10, 
        batch_size=128, 
        validation_split=0.2, 
        verbose=1
    )
    
    print("\nTraining VGG16 model...")
    # VGG16
    x_train_vgg = tf.image.resize(x_train[..., tf.newaxis], [32, 32]).numpy()
    x_train_vgg = np.repeat(x_train_vgg, 3, axis=-1)
    
    vgg_model = create_vgg16_model()
    history_vgg = vgg_model.fit(
        x_train_vgg, y_train, 
        epochs=5, 
        batch_size=128, 
        validation_split=0.2, 
        verbose=1
    )
    
    print("\nSaving models...")
    cnn_model.save("fashion_cnn.h5")
    vgg_model.save("fashion_vgg16.h5")
    
    with open('history_cnn.pkl', 'wb') as f:
        pickle.dump(history_cnn.history, f)
    with open('history_vgg.pkl', 'wb') as f:
        pickle.dump(history_vgg.history, f)
    
    print("Models and histories saved successfully!")
    
    print(f"\nCNN - Final accuracy: {history_cnn.history['accuracy'][-1]:.4f}, "
          f"val_accuracy: {history_cnn.history['val_accuracy'][-1]:.4f}")
    print(f"VGG16 - Final accuracy: {history_vgg.history['accuracy'][-1]:.4f}, "
          f"val_accuracy: {history_vgg.history['val_accuracy'][-1]:.4f}")

if __name__ == "__main__":
    train_and_save_models()