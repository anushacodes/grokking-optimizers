import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# a function to create a 3D gif of the training process

def create_3d_gif(history, filename):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Define the x and y range for the function
    x_min, x_max = min(history) - 2, max(history) + 2
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = np.linspace(x_min, x_max, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Define the function (assuming f(x, y) = f(x) + f(y))
    Z = function(X) + function(Y)

    # Scatter plot of optimization path
    history_x = np.array(history)
    history_y = np.array(history) 
    history_z = function(history_x) + function(history_y)

    def update(i):
        ax.clear()
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.scatter(history_x[:i], history_y[:i], history_z[:i], color='red', s=40)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x, y)")
        ax.set_title(f"Optimizer Path - Iteration {i+1}")

    ani = animation.FuncAnimation(fig, update, frames=len(history), repeat=False)
    ani.save(filename, writer="pillow", fps=10)
    plt.close(fig)

    print(f"3D GIF saved as {filename}")


# a function to create a 2D gif of the training process
def create_gif(history, filename):
    fig, ax = plt.subplots(figsize=(8, 5))

    x_min, x_max = min(history) - 10, max(history) + 10 
    x_vals = np.linspace(x_min, x_max, 500)
    y_vals = function(x_vals)

    y_min, y_max = np.min(y_vals), np.max(y_vals)
    
    y_margin = (y_max - y_min) * 0.2  # 20% margin
    y_min -= y_margin
    y_max += y_margin

    def update(i):
        ax.clear()
        ax.plot(x_vals, y_vals, label="Function f(x)", color="blue", linewidth=2)
        ax.scatter(history[i], function(history[i]), color="red", s=100, label="Current x") 
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(f"Gradient Descent Progress (Iteration {i+1})")
        ax.legend()
        ax.grid(True)

    ani = animation.FuncAnimation(fig, update, frames=len(history), repeat=False)
    ani.save(filename, writer="pillow", fps=10) 
    plt.close(fig)  

    print(f"GIF saved as {filename}")


def function(x):
    return np.sin(3 * x) + 0.5 * x**2

def function_grad(x):
    return 3 * np.cos(3 * x) + x

# for sgd with momentum
def function2(x):
    return x**2 + 3*x + 5 

def function_grad2(x):
    return 2*x + 3 

# for adagrad
def function3(x):
    return x**2 + 3*np.sin(x)

def function_grad3(x):
    return 2*x + 3*np.cos(x)  # Derivative of f(x)
