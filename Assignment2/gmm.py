
# Group Members: Na Li, Yash Naik
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")


# Establish Gaussian Probability Model
def Gaussian(data, mean, cov):
    # normalize the probability
    d = np.shape(cov)[0]
    cov_determinant = np.linalg.det(cov)
    cov_inverse = np.linalg.inv(cov)
    x = -0.5 * np.dot(np.dot((data - mean).T, cov_inverse), (data - mean))
    normal_distribution = (1 / (np.power(2 * np.pi, d / 2))) * np.power(cov_determinant, -0.5) * np.exp(x)

    return normal_distribution


def E_step(data, K_clusters, mean, cov, amp):
    num_points = data.shape[1]
    E_weight = np.zeros((num_points, K_clusters))

    for i in range(num_points):
        top = [amp[c] * Gaussian(data[:, i], mean[c], cov[c]) for c in range(K_clusters)]
        bottom = np.sum(top)
        for c in range(K_clusters):
            E_weight[i][c] = top[c] / bottom

    return E_weight


def M_step(data, K_clusters, weight):
    mean = []
    cov = []
    amp = []
    num_points = data.shape[1]
    # calculate the total weight of each cluster, and mean, covariance, and amplitude
    for c in range(K_clusters):
        Nc = np.sum(weight[i][c] for i in range(num_points))
        temp_mean = (1.0 / Nc) * np.sum([weight[i][c] * data[:, i] for i in range(num_points)], axis=0)
        mean.append(temp_mean)
        temp_cov = (1.0 / Nc) * np.sum(
            [weight[i][c] * (data[:, i] - mean[c]).reshape(2, 1) * (data[:, i] - mean[c]).reshape(2, 1).T for i in
             range(num_points)], axis=0)
        cov.append(temp_cov)
        amp.append(Nc / num_points)

    return mean, cov, amp


def GMM(data, K_clusters):
    num_points = data.shape[1]
    d = data.shape[0]

    prob = np.zeros((num_points, K_clusters))

    # start with random probability
    for i in range(num_points):
        prob[i] = np.random.dirichlet(np.ones(3), size=1)

    iteration = 0
    while True:
        mean, cov, amp = M_step(data, K_clusters, prob)
        new_prob = E_step(data, K_clusters, mean, cov, amp)
        # the condition to terminate the loops
        if (np.abs(new_prob - prob) < 0.000001).all() or iteration > 500:
            break

        iteration += 1
        prob = new_prob.copy()

    return mean, cov, amp, iteration


dataset = []
f = open('clusters.txt')
for row in f:
    row = row.strip().split(',')
    for i in range(len(row)):
        row[i] = float(row[i])
    dataset.append(row)

dataset = np.mat(dataset).T
mean, covariance, amplitude, counts = GMM(dataset, 3)

print('Means of the three Gaussian distributions: ', mean)
print('Covariance of the three Gaussian distributions: ', covariance)
print('Amplitude of the three Gaussian distributions: ', amplitude)

