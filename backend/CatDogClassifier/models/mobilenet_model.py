import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model
def build_mobilenet_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False)

    # Freeze the base model
    base_model.trainable = False

    # Add new model layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    return model

# Load the dataset
num_classes = 2
model = build_mobilenet_model(num_classes)
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

train_data_dir = './data/train'

# Define the data generators
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Train the model
model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=100
)


model_mobilenet = build_mobilenet_model(num_classes)
model_mobilenet.summary()

