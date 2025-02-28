import numpy as np
from visualisation import create_3d_gif, create_gif, function, function_grad

"""

Stochastic Gradient Descent

It approximates the gradient using a small subset of data (or a single sample) in each iteration.
  - Same update rule as standard gradient descent, but ∇Jᵢ(θₜ) is the gradient computed on a random sample i or a mini-batch.
  - This introduces noise but allows for faster iterations and can help escape local minima.

  
  Parameters: (same as gd)
    - lr: learning rate
    - initial_x: starting point for gradient descent
    - iters: maximum number of iterations
    - eps: convergence threshold

    Returns:
    - x: approximate local minimum found
    - history: list of x values at each iteration

"""


def stochasticGD(lr, initial_x, iters, eps = 1e-4):
  
  x = initial_x
  history = []

  for i in range(iters):
      # to introduce randomness (simulate noisy gradients)
      noise = np.random.normal(0, 0.1)  # small gaussian noise
      gradient = function_grad(x) + noise 

      x_new = x - lr * gradient
      history.append(x_new) 

      # check for convergence
      if abs(x_new - x) < eps:
          print(f"Converged after {i+1} iterations.")
          print(f"Iteration {i+1}: x = {x_new}, f(x) = {function(x_new)}")
          return x_new, history
      
      x = x_new  

  print(f"Iteration {iters}: x = {x}, f(x) = {function(x)}")

  return x, history



sgd_x, sgd_history = stochasticGD(lr = 0.1, initial_x = 5, iters = 200)

filename = "sgd.gif"

create_gif(sgd_history, filename)
print("GIF saved as 'sgd.gif'")

create_3d_gif(sgd_history, "sgd_3d.gif")
