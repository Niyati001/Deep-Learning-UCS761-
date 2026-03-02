import pandas as pd
import numpy as np

# STEP1: Load & Inspect the dataset
data= pd.read_csv('multiple_linear_regression_dataset.csv')

print(data.head(5))
print(data.columns)
print(data.shape)

#  There are 100 rows and 3 columns: age, experience, and income.

#STEP 2: Separate Inputs and Output
X = data[["age", "experience"]]
y = data["income"]

# I notice:
# X has 2 columns because we have 2 input features.
# y has 1 column because salary is the single target value

#STEP3: Initialize weights and bias
n_features= X.shape[1] 
w= np.zeros(n_features) 
b= 0.0

print("Initial weights:", w)
print("Initial bias:", b)

# Im starting from zeros
# Earlier I thought random values may be better
#but small values prevent instability

#STEP4: Forward Pass
def predict(X,w,b):
    return X.dot(w) + b

# There is no activation function.
# This is pure linear regression.
# Output can be any real number.

#STEP5: MSE Loss
def mean_squared_error(y, y_hat):
    return ((y_hat-y)**2).mean()

#STEP6: Gradient Descent
def compute_gradients(X,y,y_hat):
    N= len(y)
    dw= (2/N) * X.T.dot(y_hat-y)
    db= (2/N) * np.sum(y_hat-y)
    return dw, db

#STEP7: Update rule
def update_parameters(w,b,dw,db,lr):
    w= w - lr*dw
    b= b - lr*db
    return w,b


lr= 0.0001
epochs= 1000

for epoch in range(epochs):
    y_hat= predict(X,w,b)
    loss= mean_squared_error(y,y_hat)
    dw, db= compute_gradients(X,y,y_hat)
    w,b= update_parameters(w,b,dw,db,lr)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# I expect loss to decrease gradually.
# If loss increases, learning rate may be too high.


print("Final weights:", w)
print("Final bias:", b)

new_candidate = np.array([4.5, 68])
predicted_salary = new_candidate.dot(w) + b

print("Predicted salary:", predicted_salary)

# I expect prediction to be within realistic salary range.
# If it is extremely large or negative, something is wrong.



# At epoch 0, loss was extremely high.
# I realize this happens because weights started at zero,
# so predictions were far from actual income values.

# As epochs increased, loss decreased steadily.
# This shows gradient descent is moving in the correct direction.

# I notice experience has a larger weight than age.
# This suggests experience contributes more to income than age.

# Earlier I thought large initial loss meant something was wrong,
# but now I understand it is expected with random initialization.