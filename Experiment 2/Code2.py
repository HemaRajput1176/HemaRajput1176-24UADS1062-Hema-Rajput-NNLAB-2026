%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML, display

# XOR Data
inputs = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

targets = np.array([[0],[1],[1],[0]])

np.random.seed(42)

# Activation Function
def activation(x):
    val = 1/(1+np.exp(-x))
    return val

def activation_deriv(a):
    return a*(1-a)

input_dim = 2
hidden_dim = 5
output_dim = 1

learning_rate = 0.1
num_epochs = 25

weights_input_hidden = np.random.randn(input_dim, hidden_dim)
bias_hidden = np.zeros((1,hidden_dim))

weights_hidden_output = np.random.randn(hidden_dim, output_dim)
bias_output = np.zeros((1,output_dim))

loss_history = []
epoch_predictions = []
weights_over_time = []

# Training function
for epoch in range(num_epochs):
    print(f"\n EPOCH {epoch+1}")

    # Forward
    hidden_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    print("Hidden Input:\n", hidden_input)

    hidden_output = activation(hidden_input)
    print("Hidden Output:\n", hidden_output)

    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    print("Output Input:\n", output_input)

    predictions = activation(output_input)
    print("Predicted Output:\n", predictions)
    epoch_predictions.append(predictions.copy())

    # Loss
    loss = np.mean((targets - predictions)**2)
    loss_history.append(loss)
    print("LOSS:", loss)

    # Accuracy
    accuracy = np.mean((predictions > 0.5) == targets)
    print("ACCURACY:", accuracy)

    # Backprop
    output_error = (predictions - targets) * activation_deriv(predictions)

    grad_weights_hidden_output = np.dot(hidden_output.T, output_error)
    grad_bias_output = np.sum(output_error, axis=0, keepdims=True)

    hidden_error = np.dot(output_error, weights_hidden_output.T)
    hidden_grad = hidden_error * activation_deriv(hidden_output)

    grad_weights_input_hidden = np.dot(inputs.T, hidden_grad)
    grad_bias_hidden = np.sum(hidden_grad, axis=0, keepdims=True)

    # Update
    weights_input_hidden -= learning_rate * grad_weights_input_hidden
    bias_hidden -= learning_rate * grad_bias_hidden
    weights_hidden_output -= learning_rate * grad_weights_hidden_output
    bias_output -= learning_rate * grad_bias_output

    print("Updated Weights Input-Hidden:\n", weights_input_hidden)
    print("Updated Weights Hidden-Output:\n", weights_hidden_output)

    # Save weights for animation
    weights_over_time.append((weights_input_hidden.copy(), bias_hidden.copy(), weights_hidden_output.copy(), bias_output.copy()))

# Graph for each epochs
for i, pred in enumerate(epoch_predictions):
    plt.figure(figsize=(5,4))

    plt.scatter(range(4), targets, label="True", s=120, color='orange')
    plt.scatter(range(4), pred, label="Predicted", s=120, color='green')

    plt.title(f"Epoch {i+1}")
    plt.ylim(-0.2,1.2)
    plt.legend()
    plt.show()

# Loss Graph
plt.figure(figsize=(7,5))
plt.plot(loss_history, marker='o', color='green')
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Grid function for surface
def compute_surface(w_ih, b_h, w_ho, b_o):
    grid_x, grid_y = np.meshgrid(
        np.linspace(0,1,30),
        np.linspace(0,1,30)
    )

    grid = np.c_[grid_x.ravel(), grid_y.ravel()]

    hidden = activation(np.dot(grid, w_ih) + b_h)
    preds = activation(np.dot(hidden, w_ho) + b_o)

    return grid_x, grid_y, preds.reshape(grid_x.shape)

# 3D animation
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')

def update_plot(i):
    ax.clear()
    w_ih, b_h, w_ho, b_o = weights_over_time[i]
    gx, gy, gz = compute_surface(w_ih, b_h, w_ho, b_o)

    ax.plot_surface(gx, gy, gz, alpha=0.7, color='cyan')
    ax.scatter(inputs[:,0], inputs[:,1], targets[:,0], s=120, color='magenta')

    ax.set_title(f"Learning XOR — Epoch {i+1}")
    ax.set_zlim(0,1)

    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    ax.set_zlabel("Output")

ani = FuncAnimation(
    fig,
    update_plot,
    frames=len(weights_over_time),
    interval=250
)

plt.close(fig)

display(HTML(ani.to_jshtml()))

# Final 3D surface
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

gx, gy, gz = compute_surface(weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)

ax.plot_surface(gx, gy, gz, alpha=0.8, color='pink')
ax.scatter(inputs[:,0], inputs[:,1], targets[:,0], s=120, color='purple')

ax.set_title("Final Learned XOR Surface")
ax.set_zlim(0,1)

plt.show()
