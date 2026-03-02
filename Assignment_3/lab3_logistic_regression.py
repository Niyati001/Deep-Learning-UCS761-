import pandas as pd
import numpy as np

df= pd.read_csv('glass.csv')

print(df.head())
print(df.columns)
print(df.shape)

df['y']= (df["Type"]==1).astype(int)
df= df.drop(columns= ['Type'])

# I am converting multi-class classification into binary.
# Earlier I thought logistic regression works for any number of classes,
# but now I understand we are solving a binary case first.

X= df.drop(columns= ["y"]).values
y= df["y"].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)

# I realize scaling prevents large feature values
# from dominating the gradient updates.

def sigmoid(z):
    return 1/(1+ np.exp(-z))
    
def predict_proba(X, w, b):
    z= X@w + b
    p= sigmoid(z)
    return p

def loss(y,p):
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

# Cross-entropy punishes confident wrong predictions heavily.
# This is different from MSE, which only measures numeric distance.

def update_weights(X,y,w,b,lr):
    p= predict_proba(X,w,b)
    error= p-y

    w= w- lr* X.T @ error/ len(y)
    b= b- lr* np.mean(error)
    return w,b

w = np.zeros(X_train.shape[1])
b = 0.0
lr = 0.1
epochs = 200

for epoch in range(epochs):
    w, b = update_weights(X_train, y_train, w, b, lr)

    if epoch % 20 == 0:
        p = predict_proba(X_train, w, b)
        print("Loss:", loss(y_train, p))

def predict_label(p, threshold=0.5):
    return (p >= threshold).astype(int)

# Higher threshold makes the model more conservative.
# This is useful when false positives are costly.


# ."Logistic regression differs from the perceptron in how it interprets 
# the output of the linear function. While both models compute the same linear score
# z=w⋅x+b, the perceptron applies a hard step function that produces only 0 or 1,
# making an abrupt decision. Logistic regression instead applies the sigmoid function, 
# which converts the score into a probability between 0 and 1. The sigmoid matters because 
# it preserves uncertainty — it tells us how confident the model is, not just the final class.
# This also allows the use of cross-entropy loss, which strongly penalizes confident wrong 
# predictions and enables smooth gradient-based learning.
# However, one major problem still remains unsolved: the decision boundary is still linear
# meaning the model cannot correctly classify data that is not linearly separable.