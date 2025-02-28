import numpy as np
from visualisation import create_3d_gif, create_gif, function3, function_grad3

"""

AdaGrad
    - Adds to SGD by introducing learning rate as a parameter, gets modified according to previous gradients for each parameter
    - If the previous gradient is large, the learning rate gets smaller. 
        - However this causes a problem in later step as th elearning rate becomes too small as gradient keeps accumulating.
    - Useful for sparse data

"""

def AdaGrad(lr, initial_x, iters, eps = 1e-4):
  g = 0    # running sum of squared gradients (lr)
  x = initial_x  
  history = [x]
  new_lr = 0

  for i in range(iters):
    gradient = function_grad3(x)
    g += gradient ** 2

    new_lr = lr / (np.sqrt(g) + 1e-8)

    x -= new_lr * gradient

    history.append(x)

    # Convergence check
    if abs(gradient) < eps:
        print(f"Converged after {i+1} iterations.")
        print(f"Final: x = {x}, f(x) = {function3(x)}")
        return x, history

  print(f"Max iterations reached ({iters}).")
  print(f"Final: x = {x}, f(x) = {function3(x)}")
  return x, history



x_adagrad, adagrad_history = AdaGrad(lr = 0.9, initial_x = 4, iters = 100)

filename = "adagrad.gif"
create_gif(adagrad_history, filename)

create_3d_gif(adagrad_history, "adagrad_3d.gif")
