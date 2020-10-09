# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("fastmap-data.txt", header=None, delimiter='\t', index_col=None)
data.head(10)

d = np.array(data)

labels = pd.read_csv("fastmap-wordlist.txt", header=None)
labels.head(10)

d = sorted(d, key=lambda x: x[2], reverse=True)
d

dist_1 = int(d[0][2])
dist_1

result = [[] for i in range(10)]


def f_map(k, dist_1, d):
    if k <= 0:
        return

    a = int(d[0][0])  # object a
    b = int(d[0][1])  # object b
    d_ab = dist_1  # distance between a and b

    for i in range(1, 11):
        d_ai, d_bi = 0, 0  # dist. b/w 'a','b' and random object i
        for x in d:
            if (x[0] == i and x[1] == a) or (x[1] == i and x[0] == a):
                d_ai = x[2]
                # print("x[2]: ",x[2])
                # print('\n')
                # print("d_ai: ",d_ai)
            if (x[0] == i and x[1] == b) or (x[1] == i and x[0] == b):
                d_bi = x[2]
                # print("x[2]: ", x[2])
                # print('\n')
                # print("d_bi: ",d_bi)

        x_i = (d_ai ** 2 + d_ab ** 2 - d_bi ** 2) / (2 * d_ab)  # calc. x coordinate for object i
        # print("x_i:", x_i)
        result[i - 1].append(x_i)  # add x coordinate to list of coordinates
    print("list: ", result)

    d_new = []

    for obj_i in d:
        dist_bar = obj_i[2] ** 2 - (result[obj_i[0] - 1][0] - result[obj_i[1] - 1][0]) ** 2  # New distance formula
        # print("D-prime:",dist_bar)
        if dist_bar > 0.0:
            dist_bar = np.sqrt(dist_bar)

        d_new.append([obj_i[0], obj_i[1], dist_bar])
    # print('\n')
    # print("New_D: ",d_new)
    # print('\n')

    d_new = sorted(d_new, key=lambda x: x[2], reverse=True)
    # print("After sorting new array \n-")
    # print(d_new)

    # print("New_D[0][2]: ", d_new[0][2])
    # print('\n')

    # print("round 1 is complete!!")

    f_map(k - 1, int(d_new[0][2]), d_new)


f_map(2, dist_1, d)

label = labels.values.tolist()

for i in range(10):
    plt.plot(result[i][0], result[i][1], 'or')
    plt.annotate(label[i], xy=(result[i][0], result[i][1]))
plt.show()
