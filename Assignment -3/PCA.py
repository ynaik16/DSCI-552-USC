import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("pca-data.txt", delimiter = '\t', header = None)
data.head(10)

X = np.array(data)
X

plt.scatter(X[:, 0],X[: ,1], cmap = plt.cm.get_cmap('viridis'))
plt.plot([X[:, 0].min(), X[:, 0].max()], [X[: ,1].max(), X[: ,1].min()], color = 'r')
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt. colorbar()
plt.show()

mean = np.mean(X, axis =0)
#print(mean)

X_bar = X - mean
X_bar

cov_mat = np.dot(X_bar.T,X_bar)/len(X)
cov_mat

eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

print("EigenValues: \n", eigenvalues)
print('\n')
print('EigenVectors: \n', eigenvectors)

sort = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sort]
eigenvectors = eigenvectors[:,sort]

new_eigenvector = eigenvectors[:,:2]
new_eigenvector

z = np.dot(X, new_eigenvector)
z

a1 = z[:,0]
a2 = z[:,1]

plt.scatter(a1,a2, cmap = plt.cm.get_cmap('viridis',2))
plt.plot([a1.min(), a1.max()], [0, 0], color = 'r')           # PC 1
plt.plot([0, 0], [a2.min(), a2.max()], color = 'm')           # PC 2
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt. colorbar()
plt.show()