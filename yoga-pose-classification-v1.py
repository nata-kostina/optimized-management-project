from dotenv import load_dotenv
from memory_profiler import profile
import os
import torch
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.applications.vgg16 import VGG16
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from time import perf_counter

load_dotenv()

# -------------------------------
# Load pre-saved tensors
# -------------------------------
@profile
def load_tensors():
    tensor_root = os.getenv("TENSOR_DIR")
    
    X_train = torch.load(os.path.join(tensor_root, "train_X.pt"))
    y_train = torch.load(os.path.join(tensor_root, "train_y.pt"))
    
    X_val = torch.load(os.path.join(tensor_root, "test_X.pt"))
    y_val = torch.load(os.path.join(tensor_root, "test_y.pt"))
    
    # Convert to numpy for TensorFlow
    X_train = X_train.permute(0,2,3,1).numpy()
    X_val   = X_val.permute(0,2,3,1).numpy()
    
    return X_train, y_train, X_val, y_val

X_train, y_train, X_val, y_val = load_tensors()

# One-hot encode labels
no_classes = len(set(y_train.tolist()))
y_train = tf.keras.utils.to_categorical(y_train, no_classes)
y_val   = tf.keras.utils.to_categorical(y_val, no_classes)

# -------------------------------
# Create tf.data.Dataset
# -------------------------------
batch_size = 16
EPOCHS = 1

# Training dataset (shuffle + batch)
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(buffer_size=1000).batch(batch_size)

# Validation dataset (batch only, no shuffle, no augmentation)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(batch_size)

# -------------------------------
# Build VGG16-based model
# -------------------------------
model_name = "Yoga-Pose-Classification"

vgg16 = VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(224,224,3),
    pooling=None
)

# Freeze VGG16 layers
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

# Compile
model_vgg.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(0.001),
    metrics=['accuracy']
)

# -------------------------------
# ReduceLROnPlateau callback
# -------------------------------
lr_reduction = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=3,
    verbose=1,
    factor=0.5,
    min_lr=0.00001
)

cbs = [lr_reduction]

# -------------------------------
# Train the model
# -------------------------------
t0 = perf_counter()   
history = model_vgg.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cbs
)
t1 = perf_counter() 

t2 = perf_counter()   
# Evaluate on training dataset
train_loss, train_acc = model_vgg.evaluate(train_ds)
# Evaluate on validation dataset
test_loss, test_acc = model_vgg.evaluate(val_ds)
t3 = perf_counter() 

print(f"Final Train Accuracy: {train_acc * 100:.2f}%")
print(f"Final Validation Accuracy: {test_acc * 100:.2f}%")

print(f"Training time: {t1 - t0:.6f} seconds")
print(f"Evaluation time: {t3 - t2:.6f} seconds")
