import tensorflow as tf
tf.compat.v1.losses.sparse_softmax_cross_entropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# for cheanging image Size and properties use image data generator
train_datagen = ImageDataGenerator(
                                   rescale=1./255, 
                                   rotation_range=20, # Rotate the image
                                   zoom_range=0.2, # Zoom the image
                                   horizontal_flip=True, # Flip the image
                                   fill_mode='nearest') # Fill the missing pixels


# Let's assume that the training images are in the 'train' folder
train_generator = train_datagen.flow_from_directory(
    './data/train', # This is the source directory for training images
    target_size=(150, 150), # Change the size of images to 150x150
    batch_size=32, # Number of images to be processed at once
    class_mode='binary' # Since we use binary_crossentropy loss, we need binary labels
)

# Validation data should not be augmented
validation_generator = ImageDataGenerator(rescale=1./255)

# Let's assume that the validation images are in the 'validation' folder
validation_generator = validation_generator.flow_from_directory(
    './data/validation', # This is the source directory for validation images
    target_size=(150, 150), # Change the size of images to 150x150
    batch_size=32,          # Number of images to be processed at once
    class_mode='binary'     # Since we use binary_crossentropy loss, we need binary labels
)
