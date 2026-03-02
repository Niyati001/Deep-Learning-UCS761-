Assignment 4 – Multiple Linear Regression using Linear Perceptron

Objective:
In this lab, I implemented Multiple Linear Regression using a single linear neuron (perceptron without activation).
The goal was to understand:
-How numeric prediction differs from classification
-How loss replaces correctness
-How gradients guide parameter updates
-Why this is still the same perceptron at its core

Dataset:
The dataset contains:
age
experience
income (target variable)

Shape of dataset:
(20 samples, 3 columns)

Inputs:
age
experience

Output:
income

Model Structure
The model used: y^​=w1​x1​+w2​x2​+b

There is:
No activation function
No threshold
No sigmoid
This is pure linear regression.

Loss Function:
Mean Squared Error (MSE) was used: L=N1​∑(y−y^​)2

Why MSE?
Output is continuous (income)
Large mistakes are penalized more
Suitable for regression problems

I now understand that MSE cares about numeric distance, not classification confidence.
Gradient Descent Learning

Parameters were updated using:
w:=w−η⋅dw
b:=b−η⋅db

Where gradients are:
dw=2/N​XT(y^​−y)
db=2/N​∑(y^​−y)

I observed that:
Large error → large gradient → bigger updates
Small error → small gradient → smaller updates
This matches the intuition behind gradient descent.

Training Observations
Initial Loss: Very large
Final Loss: Significantly reduced
The loss decreased steadily across epochs, which indicates:
Learning rate was stable
Gradients were computed correctly
Model was converging

Earlier, I thought a very large initial loss meant something was wrong, but I now understand it is expected when weights start at zero.

Learned Parameters (Interpretation)
Final weights showed:
Experience has a larger coefficient than age
This suggests experience contributes more strongly to income
This aligns with real-world intuition.

Key Learning Reflection
Earlier, I viewed the perceptron mainly as a classification model.

Now I understand:
The core structure is the same.
Only the loss and output interpretation changed.
Removing activation does not remove learning.
Gradient is the mechanism that truly drives learning.

This lab helped me see that:

The model equation stays the same.
The loss defines the task.
The gradient defines the learning direction.

Final Conclusion
This lab demonstrated that:
Linear regression is simply a perceptron without activation.
Learning is guided entirely by gradient descent.
Loss function choice depends on the problem type.
Understanding the reasoning process is more important than just reaching the final answer.