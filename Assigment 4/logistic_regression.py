# Yash Naik, Na Li

import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("classification.txt", header = None, delimiter = ',')
df.head()

df = df.drop(columns=[3], axis = 1)
df.head()

X_train = df.iloc[:,:3]
X_train

y_train = df[4]
y_train

sns.countplot(x=df[4],data= df)

weights = np.random.rand(1, X_train.shape[1]+1)                 # d+1
print("Initial weights: ",weights)
lr = 0.01

Y = y_train[:,np.newaxis]
#print(Y)
padding = np.ones((X_train.shape[0], 1))
X = np.concatenate((X_train, padding), axis = 1)
#print(X)

def calc_wt(weights, X_train, y_train,lr):
#    w = np.array(weights)
    for i in range(len(X_train)):
        #x_i = X_train[i]
        #y_i = y_train[i]

        linear_eq = np.multiply(np.dot(X_train, weights.T),y_train)
        sig = -1 * np.exp(-linear_eq)/(1+ np.exp(-linear_eq))
        sig = np.multiply(sig, np.multiply(X_train, y_train))
        sig = np.sum(sig, axis =0)
        sig = sig/X_train.shape[0]

        weights -= lr * sig

        #print(y_i)
        #sig = (1/(1+ np.exp(-(y_i * w * x_i.T))))
        #weights=(weights - (lr*sig*y_i*x_i)/len(X_train))

        return weights
for i in range(7000):
    wt = calc_wt(weights, X, Y, lr)

print("\nFinal weights: ",wt)


def predict_label(X, w):
    predict = np.dot(X, w)
    predict[predict >= 0] = 1
    predict[predict < 0] = -1

    return predict

predicted = predict_label(X, wt.T)
misclassified = len(np.where(predicted != Y)[0])
total = X.shape[0]
accuracy = 1 - misclassified / total
print('After {} iterations,'.format(7000) + ' the accuracy of Logistic Rgression is : {}'.format(accuracy))