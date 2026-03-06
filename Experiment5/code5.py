import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape data for CNN
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

print("Training Data Shape:", x_train.shape)
print("Testing Data Shape:", x_test.shape)

# Function to build CNN model
def create_model(filter_size, optimizer_name):

    model = Sequential([
        
        Input(shape=(28,28,1)),
        
        Conv2D(32, (filter_size,filter_size), activation='relu'),
        MaxPooling2D((2,2)),
        
        Conv2D(64, (filter_size,filter_size), activation='relu'),
        MaxPooling2D((2,2)),
        
        Flatten(),
        
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=optimizer_name,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Experiment Parameters
filter_sizes = [3,5]
batch_sizes = [32,64]
optimizers = ['adam','sgd']


# Run experiments
for f in filter_sizes:
    for b in batch_sizes:
        for opt in optimizers:

            print("\n-----------------------------------")
            print("Filter Size:",f," Batch Size:",b," Optimizer:",opt)
            print("-----------------------------------")

            model = create_model(f,opt)

            history = model.fit(
                x_train,
                y_train,
                epochs=5,
                batch_size=b,
                validation_split=0.2,
                verbose=1
            )

            loss, accuracy = model.evaluate(x_test,y_test)

            print("Test Accuracy:",accuracy)


            # =========================
            # Graph 1 : LOSS vs EPOCH
            # =========================

            plt.figure()

            plt.plot(history.history['loss'], label="Training Loss")
            plt.plot(history.history['val_loss'], label="Validation Loss")

            plt.title(f"Loss Graph (Filter {f}, Batch {b}, Opt {opt})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")

            plt.legend()
            plt.show()


            # =========================
            # Graph 2 : ACCURACY vs EPOCH
            # =========================

            plt.figure()

            plt.plot(history.history['accuracy'], label="Training Accuracy")
            plt.plot(history.history['val_accuracy'], label="Validation Accuracy")

            plt.title(f"Accuracy Graph (Filter {f}, Batch {b}, Opt {opt})")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")

            plt.legend()
            plt.show()
