import tensorflow as tf
from models.model import build_model  # Importing the first model
from models.model_2 import build_model_2  # Importing the second model
from models.model_3 import build_model_3  # Importing the third model
from models.model_vgg16 import build_VGG16_model_categorical  # Importing the VGG16 model
from keras.preprocessing.image import ImageDataGenerator

def train_model(data_dir, model_name):
    # Choose the model based on the model name
    if model_name == 'model_1':
        model = build_model()
    elif model_name == 'model_2':
        model = build_model_2()
    elif model_name == 'model_3':
        model = build_model_3()
    elif model_name == 'model_vgg16':
        model = build_VGG16_model_categorical()
    else:
        raise ValueError("Invalid model name")

    # Initialize data generators for training and validation
    generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Prepare training data
    train_data = generator.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Prepare validation data
    validation_data = generator.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Train the model
    history = model.fit(
        train_data,
        epochs=10,
        validation_data=validation_data
    )

    # Evaluate the model
    score = model.evaluate(validation_data, verbose=0)
    print(f'Test loss: {score[0]}')
    print(f'Test accuracy: {score[1]}')

if __name__ == '__main__':
    # Define the data directory
    data_dir = './data'
    # Train model 2 as an example
    train_model(data_dir, 'model_2')
