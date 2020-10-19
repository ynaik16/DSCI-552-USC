# Na Li, Yash Naik

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tqdm import tqdm


# In[2]:


def predict_label(X,w):
    
    predict = np.dot(X,w)
    predict[predict >= 0] = 1
    predict[predict < 0] = -1
    
    return predict


# In[3]:


def perceptron_learning(X, y, learning_rate, epochs):
    
    N, d = X.shape
    weight = np.random.random(d + 1)
    ones = np.ones(N).reshape(N, 1)
    X = np.concatenate((X, ones), axis = 1)

    for i in tqdm(range(epochs)):
        for i in range(N):
            if np.dot(weight, X[i]) < 0 and y[i] == 1:
                weight += learning_rate * X[i]
            elif np.dot(weight, X[i]) >= 0 and y[i] == -1:
                weight -= learning_rate * X[i]
    
    return X, weight


# In[4]:


data = np.loadtxt('classification.txt', dtype = "float", delimiter = ",")
X = data[:,0:3]
y = data[:,3]
epochs = 800
X, weight = perceptron_learning(X, y, 0.01, epochs)
print('The weight is: {}'.format(weight))


predicted = predict_label(X, weight)
misclassified = len(np.where(predicted != y)[0])
total = X.shape[0]
accuracy = 1 - misclassified / total
print('After {} times iteration,'.format(epochs) + ' the accuracy of perceptron algorithm is : {}'.format(accuracy)  +  ', and there are {} misclassified instances.'.format(misclassified ))


# In[ ]:





# In[ ]:





# In[ ]:




