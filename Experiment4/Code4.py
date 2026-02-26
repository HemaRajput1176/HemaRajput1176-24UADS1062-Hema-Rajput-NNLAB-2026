# ==========================================
# THREE LAYER NEURAL NETWORK PERFORMANCE TEST
# ==========================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# ==============================
# LOAD DATASET
# ==============================

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0


# ==============================
# MODEL FUNCTION
# ==============================

def create_model(hidden1, hidden2, activation):

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(784,)),   # Correct input layer
        tf.keras.layers.Dense(hidden1, activation=activation),
        tf.keras.layers.Dense(hidden2, activation=activation),
        tf.keras.layers.Dense(10)  # output layer
    ])

    return model


# ==============================
# TRAIN FUNCTION
# ==============================

def train_model(hidden1, hidden2, activation, lr, batch_size, epochs):

    model = create_model(hidden1, hidden2, activation)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_test, y_test)
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    return history, test_acc


# ==============================
# PARAMETERS TO TEST
# ==============================

activations = ['relu', 'sigmoid', 'tanh']
hidden_sizes = [(128, 64), (256, 128)]
learning_rates = [0.001, 0.01]
batch_sizes = [128, 256]
epochs_list = [5, 10]

results = []


# ==============================
# EXPERIMENT LOOP
# ==============================

for act in activations:
    for hs in hidden_sizes:
        for lr in learning_rates:
            for bs in batch_sizes:
                for ep in epochs_list:

                    print(f"\nTraining: act={act}, hidden={hs}, lr={lr}, batch={bs}, epochs={ep}")

                    history, acc = train_model(
                        hs[0], hs[1],
                        act,
                        lr,
                        bs,
                        ep
                    )

                    results.append({
                        "activation": act,
                        "hidden": hs,
                        "lr": lr,
                        "batch": bs,
                        "epochs": ep,
                        "accuracy": acc
                    })

                    print("Accuracy:", acc)


# ==============================
# DISPLAY RESULTS
# ==============================

print("\n===== FINAL RESULTS =====")

for r in results:
    print(r)


# ==============================
# PLOT EXAMPLE TRAINING CURVE
# ==============================

plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")

plt.legend()
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()
