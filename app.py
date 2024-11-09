import streamlit as st
from PIL import Image
from img import load_model
import numpy as np

st.title("Fashion MNIST Image Classifier")


def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to NumPy array and normalize
    image_array = np.array(image)
    image_array = image_array.reshape(1, 28, 28, 1)  # (1, height, width, channels)
    image_array = image_array / 255.0  # Normalize to range [0, 1]
    
    return image_array

model = load_model()

file = st.file_uploader('Upload an image of clothing',type=['png','jpeg','jpg'])

if file is not None:

    image = Image.open(file)

    st.image(image,caption='Uploaded Image')

    img_array = preprocess_image(image)


    pred = model.predict(img_array)
    pred_class = np.argmax(pred[0])

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    st.write(f"It is an : {class_names[pred_class]}")
