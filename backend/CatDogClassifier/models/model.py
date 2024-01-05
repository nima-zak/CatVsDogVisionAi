import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)), 
    MaxPooling2D((2, 2)), # Reduce the size of the images

    Conv2D(64, (3, 3), activation='relu'), # Add more convolutional layers
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(), # Flatten the output of the convolutional layers
    Dense(512, activation='relu'), # Add more dense layers
    Dropout(0.5),                   # Add dropout to prevent overfitting
    Dense(1, activation='sigmoid')   # Output layer
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

def build_model():
    return model
# Print the summary
model.summary()
