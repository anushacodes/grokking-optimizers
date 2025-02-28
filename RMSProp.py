import numpy as np
from visualisation import create_3d_gif, create_gif, function3, function_grad3

def RMSProp(lr, initial_x, iters, eps = 1e-5, beta = 0.9):     # new term - beta introduced
  g = 0    # running sum of squared gradients (lr)
  x = initial_x
  history = [x]
  new_lr = 0

  for i in range(iters):
    gradient = function_grad3(x)
    # uppdated moving average of squared gradients
    g = beta * g +(1 - beta) * gradient ** 2

    new_lr = lr / (np.sqrt(g) + 1e-3) # adaptive learning rate

    x -= new_lr * gradient

    history.append(x)

    # convergence check
    if abs(gradient) < eps:
        print(f"Converged after {i+1} iterations.")
        print(f"Final: x = {x}, f(x) = {function3(x)}")
        return x, history

  print(f"Max iterations reached ({iters}).")
  print(f"Final: x = {x}, f(x) = {function3(x)}")
  return x, history



x_rmsprop, rmsprop_history = RMSProp(lr = 0.1, initial_x = 4, iters = 200)


filename = "rmsprop.gif"
create_gif(rmsprop_history, filename)

create_3d_gif(rmsprop_history, "rmsprop_3d.gif")
