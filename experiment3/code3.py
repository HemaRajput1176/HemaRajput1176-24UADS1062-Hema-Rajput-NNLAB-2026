# ==============================
# THREE LAYER NN - MNIST (FULL CORRECT VERSION)
# ==============================

# %matplotlib inline   # Uncomment if using Jupyter

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# ==============================
# 1. LOAD MNIST DATASET
# ==============================

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# ==============================
# 2. PREPROCESS DATA
# ==============================

x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test  = x_test.reshape(-1, 784).astype(np.float32) / 255.0

# FIX: Convert labels to int32
y_train = y_train.astype(np.int32)
y_test  = y_test.astype(np.int32)

# ==============================
# 3. DEFINE PARAMETERS
# ==============================

input_size = 784
hidden1 = 256
hidden2 = 128
output_size = 10

learning_rate = 0.001
epochs = 5
batch_size = 256

# ==============================
# 4. WEIGHT INITIALIZATION
# ==============================

W1 = tf.Variable(tf.random.normal([input_size, hidden1], stddev=0.1))
b1 = tf.Variable(tf.zeros([hidden1]))

W2 = tf.Variable(tf.random.normal([hidden1, hidden2], stddev=0.1))
b2 = tf.Variable(tf.zeros([hidden2]))

W3 = tf.Variable(tf.random.normal([hidden2, output_size], stddev=0.1))
b3 = tf.Variable(tf.zeros([output_size]))

# ==============================
# 5. FEED FORWARD FUNCTION
# ==============================

def forward(X):
    X = tf.cast(X, tf.float32)

    Z1 = tf.matmul(X, W1) + b1
    A1 = tf.nn.relu(Z1)

    Z2 = tf.matmul(A1, W2) + b2
    A2 = tf.nn.relu(Z2)

    Z3 = tf.matmul(A2, W3) + b3
    return Z3   # logits (NO softmax here)

# ==============================
# 6. TRAINING (BACKPROPAGATION)
# ==============================

optimizer = tf.optimizers.Adam(learning_rate)

loss_history = []
acc_history = []

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(60000).batch(batch_size)

for epoch in range(epochs):

    epoch_loss = 0
    batch_count = 0

    for x_batch, y_batch in dataset:

        with tf.GradientTape() as tape:
            logits = forward(x_batch)

            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y_batch,
                    logits=logits
                )
            )

        gradients = tape.gradient(loss, [W1, b1, W2, b2, W3, b3])
        optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2, W3, b3]))

        epoch_loss += loss.numpy()
        batch_count += 1

    epoch_loss /= batch_count
    loss_history.append(epoch_loss)

    # TEST ACCURACY
    test_logits = forward(x_test)
    predictions = tf.argmax(test_logits, axis=1)

    accuracy = tf.reduce_mean(
        tf.cast(predictions == y_test, tf.float32)
    )

    acc_history.append(accuracy.numpy())

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy.numpy():.4f}")

# ==============================
# 7. PLOT LOSS & ACCURACY
# ==============================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(acc_history)
plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.show()

# ==============================
# 8. VISUALIZE PREDICTIONS
# ==============================

indices = random.sample(range(len(x_test)), 10)

plt.figure(figsize=(12,4))

for i, idx in enumerate(indices):

    img = x_test[idx].reshape(28,28)
    pred = predictions[idx].numpy()

    plt.subplot(2,5,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Pred: {pred}")
    plt.axis('off')

plt.show()

print("Final Test Accuracy:", accuracy.numpy())
