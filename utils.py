import numpy as np
from PIL import Image

def preprocess_image(image, model_choice, input_size=28):
    image_resized = image.resize((input_size, input_size))
    img_array = np.array(image_resized).astype('float32') / 255.0
    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array
    if model_choice == "CNN":
        img_array = img_array.reshape(1, 28, 28, 1)
    else:  # VGG16
        img_array = img_array.reshape(1, 32, 32, 1)
        img_array = np.repeat(img_array, 3, axis=-1)
    return img_array
def get_input_size(model_choice):
    return 28 if model_choice == "CNN" else 32