import tensorflow as tf
from models.model import build_model  # Importing the model from model.py
from keras.preprocessing.image import ImageDataGenerator

def train_model(data_dir):
    # Load the pre-built model
    model = build_model()

    # Initialize data generators for training and validation
    generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Prepare training data
    train_data = generator.flow_from_directory(
        data_dir,
    target_size=(224, 224),                  # Resize images to 224x224
        batch_size=32,                         # Batch size
        class_mode='categorical',             # Use categorical cross-entropy loss
        subset='training',                    # Use the training subset
        shuffle=True                          # Shuffle the data
    )

    # Prepare validation data
    validation_data = generator.flow_from_directory(
        data_dir,                       # Same data directory
        target_size=(224, 224),         # Resize images to 224x224
        batch_size=32,                  # Batch size
        class_mode='categorical',        # Use categorical cross-entropy loss
        subset='validation',             # Use the validation subset
        shuffle=False                   # Do not shuffle the data
    )

    # Train the model
    history = model.fit(
        train_data,
        epochs=10,                  # Train for 10 epochs
        validation_data=validation_data          # Validate on the validation data
    )

    # Evaluate the model
    score = model.evaluate(validation_data, verbose=0)      # Evaluate the model on the validation data
    print(f'Test loss: {score[0]}')                          # Print the test loss
    print(f'Test accuracy: {score[1]}')                      # Print the test accuracy

if __name__ == '__main__':                                  # Run the code
    data_dir = './data'
    train_model(data_dir)
