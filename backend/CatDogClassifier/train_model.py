from tensorflow.keras.models import Sequential
from data.data_preprocessing import train_generator , validation_generator
from models.model import build_model

# Define the model
model = build_model()
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model use the data generator
history = model.fit(
    train_generator,                      # Use the data generator
    steps_per_epoch=100,                  # Number of images in the training set
    epochs=10,                            # Number of epochs to train the model
    validation_data=validation_generator, # Use the validation data
    validation_steps=50                   # Number of images in the validation set
    )

# Save the model
model.save('cat_dog_classifier_model.h5')

# Save the history
import pickle
with open('history.pkl', 'wb') as file:
    pickle.dump(history.history, file)