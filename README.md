# grokking-optimizers
Optimization lies at the heart of machine learning, and understanding how different algorithms perform is crucial for building efficient models. This project allowed me to dive deep into the mechanics of these algorithms, implement them from scratch, and visualize their behavior on a simple yet illustrative function.

Each algorithm was tested on a custom function, and the results were visualized using 2D and 3D animations to better understand their convergence behavior.

## Gradient Descent (GD)
Gradient Descent is the most fundamental optimization algorithm. It updates the parameters in the direction of the negative gradient, scaled by a learning rate

2D visualization            |  3D visualization
:-------------------------:|:-------------------------:
![gradient_descent](https://github.com/user-attachments/assets/e095c0db-2d10-47d5-96e4-80178a329057) | ![gradient_descent_3d](https://github.com/user-attachments/assets/81144bda-5db9-4a2c-bda5-34d4df058ae1)

Converges steadily but can be slow, especially in regions with small gradients.


## Stochastic Gradient Descent (SGD)
SGD is a variant of GD that uses a single sample (or a small batch) to estimate the gradient. This introduces noise but allows for faster iterations and can help escape local minima. The update rule is similar to GD, but the gradient is approximated using a random sample.


2D visualization            |  3D visualization
:-------------------------:|:-------------------------:
![sgd](https://github.com/user-attachments/assets/153630c9-d9a0-42aa-bdd4-a106e2490542) | ![sgd_3d](https://github.com/user-attachments/assets/a4e56fab-225d-408a-84bf-a2797a5fbf08)

Converges faster but with more oscillations due to the noisy gradient estimates.

## Batch Gradient Descent
Batch Gradient Descent is a compromise between GD and SGD. It uses a small batch of samples to estimate the gradient, which reduces the noise compared to SGD while still being computationally efficient.

2D visualization            |  3D visualization
:-------------------------:|:-------------------------:
![batch_gd](https://github.com/user-attachments/assets/75ffa349-3ede-4f45-ba24-db1e43f31a88) | ![batch_gd_3d](https://github.com/user-attachments/assets/9c643e0c-9818-49c6-aa9e-a35c1836361a)

A balance between GD and SGD, with smoother convergence than SGD but faster than GD.

## SGD with Momentum

SGD with Momentum accelerates convergence by adding a fraction of the previous update to the current update. This helps in overcoming saddle points and local minima. 

2D visualization            |  3D visualization
:-------------------------:|:-------------------------:
![sgd_momentum](https://github.com/user-attachments/assets/154f1737-1f50-4cb3-8484-d56d422e45d4) | ![sgd_momentum_3d](https://github.com/user-attachments/assets/0d542ec2-423f-49d9-807d-941a3ea6314c)

Converges faster than standard SGD and is able to navigate through flat regions and saddle points more effectively.

## AdaGrad
AdaGrad adapts the learning rate based on the historical gradients. If the gradients are large, the learning rate is reduced, and vice versa. This is particularly useful for sparse data. 


2D visualization            |  3D visualization
:-------------------------:|:-------------------------:
![adagrad](https://github.com/user-attachments/assets/eacddf21-daf0-435f-86a8-b5f47a7ed69e) | ![adagrad_3d](https://github.com/user-attachments/assets/7915b48c-fcd7-4dbe-8dcf-1fdb47eb36a4)

Adapts the learning rate well, but the learning rate can become too small over time, slowing down convergence.


## RMSProp
RMSProp is an improvement over AdaGrad that uses a moving average of squared gradients to adapt the learning rate. This prevents the learning rate from decaying too quickly.

2D visualization            |  3D visualization
:-------------------------:|:-------------------------:
![rmsprop](https://github.com/user-attachments/assets/633d94dc-6b6c-42a4-b85e-5639958ee521) | ![rmsprop_3d](https://github.com/user-attachments/assets/b7ac0abf-4eda-458a-bc01-d44be5fbd155)

Improves upon AdaGrad by using a moving average of squared gradients, leading to more stable convergence.

## Adam

Adam is an adaptive optimization algorithm that combines the benefits of momentum and adaptive learning rates. It maintains two moving averages:

 - First moment (mean of gradients): This is similar to momentum and helps accelerate convergence.

 - Second moment (uncentered variance of gradients): This is used to adapt the learning rate for each parameter, similar to RMSProp.

2D visualization            |  3D visualization
:-------------------------:|:-------------------------:
![adam](https://github.com/user-attachments/assets/ca933f12-049d-4861-a5f8-6d570b0c5de9) | ![adam_3d](https://github.com/user-attachments/assets/53ffde74-3e14-406a-a4ba-e4987ec2a6fa)

Adam converges significantly faster than all other optimization methods thanks to its adaptive learning rate and momentum components. The use of momentum ensures that Adam navigates through flat regions and saddle points more effectively, resulting in a smoother optimization path.

