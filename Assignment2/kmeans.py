
# Group Members: Yash Naik, Na Li
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

df = pd.read_csv('clusters.txt', index_col=None, header=None)
df.head()

df.describe()

np.random.seed(100)
k = 3
k_centroids = {i + 1: [np.random.randint(-4, 9), np.random.randint(-4, 9)]
               for i in range(k)}

plt.scatter(df[0], df[1])
colormap = {1: 'r', 2: 'y', 3: 'g'}
for i in k_centroids.keys():
    plt.scatter(*k_centroids[i], color=colormap[i])

plt.xlim(-5, 10)
plt.ylim(-5, 10)
plt.show()

pd.set_option('display.max_columns', 8)


def assign_centers(df, centroids):  # assign all the points to their nearest centroid
    c = {'1': 'r', '2': 'y', '3': 'g'}
    for i in centroids.keys():
        df["Distance from_{}".format(i)] = np.sqrt(
            (df[0] - centroids[i][0]) ** 2 + (df[0] - centroids[i][0]) ** 2)  # distance from
        # centroids

    dist_cols = ["Distance from_{}".format(i) for i in centroids.keys()]
    df['nearest'] = df.loc[:, dist_cols].idxmin(axis=1)  # nearest centroid
    df['nearest'] = df['nearest'].map(lambda x: x.lstrip("Distance from_"))
    # print(df['nearest'])
    df['color'] = df['nearest'].map(c)
    # print(df.head(10))
    return df


df = assign_centers(df, k_centroids)

print(df.head(10))

print("_____________________________________________________________________________________________")

plt.scatter(df[0], df[1], color=df['color'])
for i in k_centroids.keys():
    plt.scatter(*k_centroids[i], color=colormap[i])

plt.xlim(-5, 10)
plt.ylim(-5, 10)
plt.show()

k_centroids

import warnings

warnings.filterwarnings("ignore")

prev_centroids = k_centroids


def update_centroids(centers):  # calculate mean of the all the points
    for i in centers.keys():  # according to resp. clusters and update centroids
        centers[i][0] = np.mean(df[df['nearest'] == str(i)][0])
        centers[i][1] = np.mean(df[df['nearest'] == str(i)][1])

    return centers


k_centroids = update_centroids(k_centroids)
# print(centroids)
plt.scatter(df[0], df[1], color=df['color'])
ax = plt.axes()
for i in k_centroids.keys():
    plt.scatter(*k_centroids[i], color=colormap[i], marker='s')
plt.xlim(-5, 10)
plt.ylim(-5, 10)
for i in prev_centroids.keys():
    x = prev_centroids[i][0]
    y = prev_centroids[i][1]
    dx = (k_centroids[i][0] - prev_centroids[i][0]) * 0.5
    dy = (k_centroids[i][1] - prev_centroids[i][0]) * 0.5
    ax.arrow(x, y, dx, dy, head_length=0.75, head_width=0.25, fc=colormap[i], ec=colormap[i])
plt.show()

while True:  # Loops the above two functions to create better clusters
    nearest_centers = df['nearest'].copy(deep=True)
    centroids = update_centroids(k_centroids)
    df = assign_centers(df, centroids)
    if nearest_centers.equals(df['nearest']):
        break

plt.scatter(df[0], df[1], color=df['color'])
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colormap[i])
plt.xlim(-5, 10)
plt.ylim(-5, 10)
plt.show()
final_centroids = centroids
print(".........................................................................")
print(final_centroids)