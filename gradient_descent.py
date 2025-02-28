import numpy as np
from visualisation import create_3d_gif, create_gif


"""

Gradient Descent

  Let θ be the parameter vector and J(θ) be the objective function:
  - Update rule: θₜ₊₁ = θₜ - η∇J(θₜ)
  - ∇J(θₜ) is the gradient of the objective function with respect to θ


Parameters:
    - lr: learning rate, which controls step size in each iteration
    - initial_x: starting point for gradient descent
    - iters: maximum number of iterations to run
    - eps: convergence threshold. Stops if step size is smaller than this value

    Returns:
    - x: approximate local minimum found
    - history: list of x values at each iteration

"""


def function(x):
    return np.sin(3 * x) + 0.5 * x**2

def function_grad(x):
    return 3 * np.cos(3 * x) + x


def gradient_descent(lr, initial_x, iters, eps = 1e-4):

# initialising x at any given point (in practice, we want to randomise this)
  x = initial_x
  history = []    

  for i in range(iters):
    gradient = function_grad(x)   # calcs gradient of a given point
    x_new = x - lr * gradient     # x_new = x_old - n(∇f(x))
    history.append(x_new)         # to store values of x for visualisation later

    # checking for convergence
    if abs(x_new - x) < eps:
        print(f"Converged after {i+1} iterations.")
        print(f"Iteration {i+1}: x = {x_new}, f(x) = {function(x_new)}")
        return x_new, history     # returns results early if converged
    
    x = x_new   # for next iter we update x_new -> x_old

  # if the loop completes without convergence, returns the last computed x
  print(f"Iteration {i+1}: x = {x}, f(x) = {function(x)}")

  return x, history


x, history = gradient_descent(0.1, 5, 200)

filename = "gradient_descent.gif"

create_gif(history, filename)
print("GIF saved as 'gradient_descent.gif'")

create_3d_gif(history, "gradient_descent_3d.gif")
