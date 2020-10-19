
#Na Li, Yash Naik

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# In[2]:


def predict_label(X,w):
    
    predict = np.dot(X,w)
    predict[predict >= 0] = 1
    predict[predict < 0] = -1
    
    return predict


# In[3]:


def pocket(X, y, learning_rate, epochs):
    
    N, d = X.shape
    weight = np.random.random(d + 1)
    ones = np.ones(N).reshape(N, 1)
    X = np.concatenate((X, ones), axis = 1)
    
    pocket_weight =  weight
    pocket_misclassified_nums = N    

    misclassified_nums_li = []

    for i in tqdm(range(epochs)):
        for i in range(N):
            if np.dot(weight, X[i]) < 0 and y[i] == 1:
                weight += learning_rate * X[i]
            elif np.dot(weight, X[i]) >= 0 and y[i] == -1:
                weight -= learning_rate * X[i]
        
        misclassified_nums = get_misclassified_nums(X, y, weight)
        misclassified_nums_li.append(get_misclassified_nums(X, y, weight))
        
        if misclassified_nums < pocket_misclassified_nums:
            pocket_misclassified_nums = misclassified_nums
    
    return X, pocket_weight, pocket_misclassified_nums, misclassified_nums_li


# In[4]:


def get_misclassified_nums(X, y, w):
    
    predict = predict_label(X, w)
    misclassified_nums = len(np.where(predict != y)[0])

    return misclassified_nums


# In[ ]:





# In[5]:


data = np.loadtxt('classification.txt', dtype = "float", delimiter = ",")
X = data[:,0:3]
y = data[:,4]

X, pocket_weight, pocket_misclassified_nums, misclassified_nums_li = pocket(X, y, 0.01, 7000)

print('The best weight after 7000 iterations : {}'.format(pocket_weight))

total_instances = X.shape[0] 
pocket_accuracy  = 1 - pocket_misclassified_nums/total_instances
print('The best accuracy after 7000 iterations: {}'.format(pocket_accuracy))

plt.xlabel('Number of iterations')
plt.ylabel('Number of misclassified points')
plt.plot(list(range(1, len(misclassified_nums_li) + 1)),misclassified_nums_li)
plt.show()



# In[ ]:





# In[ ]:





# In[ ]:




