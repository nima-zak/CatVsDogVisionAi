import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model_path, test_data_dir, target_size=(224, 224)):
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Initialize the data generators
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load the test data
    test_data = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Evaluate the model on the test data
    evaluation = model.evaluate(test_data)
    print(f' Test Loss: {evaluation[0]}')
    print(f' Test Accuracy: {evaluation[1]}')


if __name__ == '__main__':
    # Specify the model path, test data directory, and target size
    model_path = './models/model_2_trained.h5'
    test_data_dir = './test'

    # Evaluate the model
    evaluate_model(model_path, test_data_dir)