from dotenv import load_dotenv
import os
# Tensorflow Libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from keras.applications.vgg16 import VGG16
# Callbacks
from keras.callbacks import ReduceLROnPlateau
# LR Scheduler
from keras.optimizers import Adam
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

load_dotenv()

# Setting the path to the training directory that contains the data for yoga asanas
train_dir = os.getenv("TRAIN_DIR")

# Setting the path to the test directory that contains the data for yoga asanas
test_dir = os.getenv("TEST_DIR")

#  list all the files in the directory and store them in 'class_names' alphabetically
class_names = sorted(os.listdir(train_dir))
# number of classes present
no_classes = len(class_names)

# Set the batch size for training
batch_size=16

# Set the number of epochs
EPOCHS = 2

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

train_loss, train_acc = model_vgg.evaluate(train_generator)
test_loss, test_acc   = model_vgg.evaluate(validation_generator)

print(f"Final Train Accuracy: {train_acc * 100:.2f}%")
print(f"Final Validation Accuracy: {test_acc * 100:.2f}%")