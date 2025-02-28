import numpy as np
from visualisation import create_3d_gif, create_gif, function2, function_grad2

"""

SGD with Momentum
    - Speeds up SGD by adding momentum in the relevant direction to overcome saddle points and local minima.
    - The idea is to add a fraction of the previous update to the current update, 
                            thus helping to accelerate convergence and reduce oscillation.

Formula:
    - It introduce a velocity vector vₜ = βvₜ₋₁ + η∇J(θₜ)
    - Update rule: θₜ₊₁ = θₜ - ηvₜ
    - Where β is the momentum coefficient (typically 0.9)

Parameters:
    - beta: Momentum coefficient (default 0.9)

"""


def sgd_with_momentum(lr, initial_x, iters, beta = 0.9, eps = 1e-4):

    x = initial_x  
    v = 0  # initialize momentum term
    history = [x]  

    for i in range(iters):
        gradient = function_grad2(x)  
        v = beta * v + lr * gradient  # update velocity term
        x_new = x - lr * v  

        history.append(x_new)  

        # Convergence check
        if abs(x_new - x) < eps:
            print(f"Converged after {i+1} iterations.")
            print(f"Final: x = {x_new}, f(x) = {function2(x_new)}")
            return x_new, history

        x = x_new  # Update x for next iteration

    print(f"Max iterations reached ({iters}).")
    print(f"Final: x = {x}, f(x) = {function2(x)}")
    return x, history


x_opt, sgd_mom_history = sgd_with_momentum(lr = 0.1, initial_x = -4, iters = 100)

filename = "sgd_momentum.gif"
create_gif(sgd_mom_history, filename)

create_3d_gif(sgd_mom_history, "sgd_momentum_3d.gif")