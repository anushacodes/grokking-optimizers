import numpy as np
from visualisation import create_3d_gif, create_gif, function3, function_grad3

"""

Adam: combines the benefits of both Momentum and RMSProp

"""

def Adam(lr, initial_x, iters, beta1=0.9, beta2=0.999, eps=1e-8):

    x = initial_x  
    m, v = 0, 0  # initialize first and second moment
    history = [x]

    for t in range(1, iters + 1):  # starts from 1 to avoid bias correction issues
        gradient = function_grad3(x)  

        # Compute biased first and second moment estimates
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)

        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Update x
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(x)

        # Convergence check
        if abs(gradient) < eps:
            print(f"Converged after {t} iterations.")
            print(f"Final: x = {x}, f(x) = {function3(x)}")
            return x, history

    print(f"Max iterations reached ({iters}).")
    print(f"Final: x = {x}, f(x) = {function3(x)}")
    return x, history

x_adam, adam_history = Adam(lr=0.05, initial_x=4, iters=100, beta1=0.9, beta2=0.999, eps=1e-8)

filename = "adam.gif"
create_gif(adam_history, filename)

create_3d_gif(adam_history, "adam_3d.gif")


