import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model_path, test_data_dir, target_size=(224, 224)):
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Initialize the data generators
    test_datagen = ImageDataGenerator(rescale=1./255)  # Add any other preprocessing if needed

    # Load the test data
    test_data = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=target_size,
        batch_size=32,  # Consider adjusting batch size
        class_mode='categorical',
        shuffle=False
    )

    # Evaluate the model on the test data
    evaluation = model.evaluate(test_data)
    print(f' Test Loss: {evaluation[0]}')
    print(f' Test Accuracy: {evaluation[1]}')

    # Predictions and detailed report
    predictions = model.predict(test_data)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_data.classes
    class_labels = list(test_data.class_indices.keys())

    print("Classification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    print("Confusion Matrix:")
    print(confusion_matrix(true_classes, predicted_classes))

if __name__ == '__main__':
    # Specify the model path, test data directory, and target size
    model_path = './models/mobilenet_model_trained.h5'
    test_data_dir = './test'

    # Evaluate the model
    evaluate_model(model_path, test_data_dir)
