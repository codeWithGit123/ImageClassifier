import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_and_save_model(weights_path='fashion_mnist_model.weights.h5'):
    fashion = tf.keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion.load_data()
    

    X_train = X_train[..., np.newaxis] / 255.0
    X_test = X_test[..., np.newaxis] / 255.0

    model = create_model()
    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
    
    # Save the model weights
    model.save_weights(weights_path)

def load_model(weights_path='fashion_mnist_model.weights.h5'):
    model = create_model()
    model.load_weights(weights_path)
    return model

# Train and save the model if running this script directly
if __name__ == '__main__':
    train_and_save_model()