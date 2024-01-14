from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

def build_VGG16_model_categorical():
    # Load the pre-trained VGG16 model
    vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers in the VGG16 model
    for layer in vgg16_base.layers:
        layer.trainable = False

    # Add a new classifier layer on top of the VGG16 model
    model = Sequential()
    model.add(vgg16_base)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))  # 2 classes: cat and dog

    # Compile the model using 'categorical_crossentropy' as the loss function
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    return model

# Create the model
model_vgg16_categorical = build_VGG16_model_categorical()
model_vgg16_categorical.summary()
