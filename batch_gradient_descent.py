import numpy as np
from visualisation import create_3d_gif, create_gif, function, function_grad

"""

Batch Gradient Descent

    Parameters (extra):
        - batch_size: number of random samples used to approximate the gradient

"""


def batch_gradient_descent(lr, initial_x, iters, batch_size, eps = 1e-4):

    x = initial_x
    history = [x] 

    # computes batch gradient approximation by averaging gradients over batch_size samples
    for i in range(iters):
        batch_grad = np.mean([function_grad(x + np.random.uniform(-0.5, 0.5)) for _ in range(batch_size)]) 

        # updates x based on the batch gradient and learning rate
        x_new = x - lr * batch_grad  
        
        history.append(x_new)  
        
        if abs(x_new - x) < eps:
            print(f"Converged after {i+1} iterations.")
            print(f"Final: x = {x_new}, f(x) = {function(x_new)}")
            return x_new, history 
        
        x = x_new 

    print(f"Max iterations reached ({iters}).")
    print(f"Final: x = {x}, f(x) = {function(x)}")
    return x, history



batch_x, batch_history = batch_gradient_descent(lr = 0.02, initial_x = 5, iters = 201, batch_size = 5)

filename = "batch_gd.gif"

create_gif(batch_history, filename)
print("GIF saved as 'batch_gd.gif'")

create_3d_gif(batch_history, "batch_gd_3d.gif")
