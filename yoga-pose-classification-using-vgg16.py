# Import Data Science Libraries
import numpy as np
import tensorflow as tf
import os
import warnings

# Import visualization libraries
import matplotlib.pyplot as plt

# Tensorflow Libraries
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import plot_model
# Callbacks
from keras.callbacks import Callback, ReduceLROnPlateau
# LR Scheduler
from keras.optimizers import Adam
from PIL import Image, ImageFile

# %matplotlib inline
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

def image_path(image_name):
    """Returns the full path of an image given its name."""
    return os.path.join(images_dir, image_name)

def model_path(model_name):
    """Returns the full path of a model given its name."""
    return os.path.join(model_dir, model_name)
    
def plot_metric(history):
    """Plots training and validation accuracy/loss curves."""
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_accuracy) + 1)
    
    # Plot accuracy
    plt.plot(epochs, train_accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    # plt.savefig(image_path("Training and validation accuracy.png"))

    # Plot loss
    plt.figure()
    plt.plot(epochs, train_loss,label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.legend(["train_loss", 'val_loss'])

    #Save image
    # plt.savefig(image_path("Training and validation loss.png"))
    plt.show()
    
    # Setting the path to the training directory that contains the data for yoga asanas
train_dir = ""

# Setting the path to the test directory that contains the data for yoga asanas
test_dir = ""

# Setting the path to the directory containing images for the assignment
images_dir = './kaggle/working/Images'

# Setting the path to the directory where the trained models will be saved
model_dir = './kaggle/working/Models'

#  list all the files in the directory and store them in 'class_names' alphabetically
class_names = sorted(os.listdir(train_dir))
# number of classes present
no_classes = len(class_names)

print(f"Classes :{class_names}")
print(f"Total Number of Classes : {no_classes}")

class_dis_tr = [len(os.listdir(train_dir + f"/{name}")) for name in class_names]
print(class_dis_tr)

class_dis_te = [len(os.listdir(test_dir + f"/{name}")) for name in class_names]
print(class_dis_te)

# Set the batch size for training
batch_size=16

# Set the number of epochs
EPOCHS = 22

# Define the training data generator with specified augmentations
train_datagen = IDG(shear_range=0.2,      # Randomly apply shearing transformations
                    zoom_range=0.2,       # Randomly zoom inside images
                    horizontal_flip=True, # Randomly flip images horizontally
                    rescale = 1./255      # Rescale the pixel values to [0,1]
                    )

# Define the testing data generator with rescaling only
test_datagen = IDG(rescale = 1./255 )     # Rescale the pixel values to [0,1]

# Create a generator for training data from a directory
train_generator =  train_datagen.flow_from_directory(train_dir,                 # Directory path for training data
                                                    target_size = (224,224),    # Reshape images to the specified dimensions
                                                    color_mode = 'rgb',         # Color mode set to RGB
                                                    class_mode = 'categorical', # Use categorical labels
                                                    batch_size = batch_size     # Set the batch size for training
                                                     )

# Create a generator for validation data from a directory
validation_generator  = test_datagen.flow_from_directory(test_dir,              # Directory path for testing data
                                                  target_size = (224,224),
                                                  color_mode = 'rgb',
                                                  class_mode = 'categorical'
                                                 )

print("number of images", train_generator.samples)
print("number of class", len(train_generator.class_indices))
print("number of images in the classes", train_generator.samples / len(train_generator.class_indices))

print("number of images", validation_generator.samples)
print("number of class", len(validation_generator.class_indices))
print("number of images in the classes", validation_generator.samples / len(validation_generator.class_indices))
# Model Name
model_name = "Yoga-Pose-Classification"

# Load VGG16 pre-trained model
vgg16 = VGG16(include_top=False,
              weights='imagenet',
              input_shape=(224,224,3),
              pooling=None)

# Set all layers in the VGG16 model to be non-trainable
for layer in vgg16.layers:
    layer.trainable = False
    
# Define a function to create the baseline model
def create_baseline():

    # Instantiate the sequential model and add the VGG16 model:
    model_vgg = Sequential(name=model_name+"_VGG")
    model_vgg.add(vgg16)

    # Add the custom layers atop the VGG19 model:
    model_vgg.add(Flatten(name='flattened'))                                    # Flattens the input without affecting the batch size
    model_vgg.add(Dropout(0.5, name='dropout1'))                                # Apply dropout to prevent overfitting
    model_vgg.add(Dense(256, activation='relu'))                                # Add a fully connected layer with ReLU activation
    model_vgg.add(Dropout(0.2, name='dropout2'))                                # Apply another dropout
    model_vgg.add(Dense(no_classes, activation='softmax', name='predictions'))  # Add the output layer with softmax activation

    return model_vgg


# Create the VGG16-based model
model_vgg = create_baseline()

# Display a summary of the model architecture
model_vgg.summary()

model_vgg.build()

# Plot the model architecture and save it to an image file
plot_model(model_vgg)
# Plot the model architecture and save it to an image file
# plot_model(model_vgg, to_file=image_path(model_name+"_vgg.png"))

# Compile the model with specified loss function, optimizer, and metrics
model_vgg.compile(
    loss='categorical_crossentropy',# Categorical cross-entropy loss for multi-class classification
    optimizer=Adam(0.001),          # Adam optimizer with a learning rate of 0.001
    metrics=["accuracy"]            # Monitor the accuracy metric during training
)

# Create a ReduceLROnPlateau callback with the following parameters:
# This callback monitors the validation accuracy and adjusts the learning rate when the validation accuracy plateaus
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy',    # Monitors the validation accuracy to decide when to reduce the learning rate.
                                            patience=3,     # Number of epochs with no improvement after which learning rate will be reduced
                                            verbose=1,      # Verbosity level. 1 for updating messages, 0 for silence.
                                            factor=0.5,     # Learning rate will be reduced to half. New_lr = lr * factor
                                            min_lr=0.00001  # Lower bound on the learning rate. It won't reduce the learning rate below this value
                                 )

# Store the ReduceLROnPlateau callback in a list. This list can be passed to a training session.
cbs = [lr_reduction]

history = model_vgg.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=EPOCHS,
                    batch_size=batch_size,
                    callbacks=cbs,
                    shuffle=True)

plot_metric(history)
# Save the model to a file
# model_vgg.save(model_path(model_name+'_vgg16.h5'))

train_loss, train_acc = model_vgg.evaluate(train_generator)
test_loss, test_acc   = model_vgg.evaluate(validation_generator)

print(f"Final Train Accuracy: {train_acc * 100:.2f}%")
print(f"Final Validation Accuracy: {test_acc * 100:.2f}%")