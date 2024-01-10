import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Define the model
model = Sequential()                                                         
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))   # First convolutional layer , 32 filters, 3x3 kernel
model.add(MaxPooling2D((2, 2)))              # Pooling layer
model.add(Conv2D(64, (3, 3), activation='relu')) # Second convolutional layer
model.add(MaxPooling2D((2, 2)))              # Pooling layer
model.add(Flatten())                         # Flatten the output
model.add(Dense(128, activation='relu'))   # Hidden layer
model.add(Dense(2, activation='softmax'))  # 2 classes: cat and dog

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def build_model():
    return model

# Print the summary
model.summary()
