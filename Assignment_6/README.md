## Deep Model Experiment- Model D( 8 Hidden Layers)

## Observation Table:

| Activation | Final Loss | Grad Norm (L1) | Grad Norm (Last Hidden) |
| ---------- | ---------- | -------------- | ----------------------- |
| ReLU       | 0.0650     | 0.0346         | 0.00125                 |
| Sigmoid    | 1.7438     | 7.70e-06       | 2.02e-06                |

## Reflections:

- Deeper networks did not always reduce loss faster.
- Gradients in early layers were much smaller than later layers in deep networks with Sigmoid.
- Training was not equally stable for all activations.
- ReLU behaved more stable in deeper networks.
- Some models improved very slowly despite using the same learning rate.
