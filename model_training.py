# model_training.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Example usage
def train_model(frames, labels):
    frames = frames / 255.0  # Normalize frames
    X_train, X_test, y_train, y_test = train_test_split(frames, labels, test_size=0.2)
    
    model = create_model()
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    
    model.save('behavior_detection_model.h5')

# Assuming frames and labels are available
train_model(frames, labels)
