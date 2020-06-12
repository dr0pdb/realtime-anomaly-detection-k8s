import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from numpy import genfromtxt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score


def read_dataset(filePath,delimiter=','):
    data = pd.read_csv(filePath) 
    return data


def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma


def estimateGaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma


def multivariateGaussian(dataset,mu,sigma):
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.pdf(dataset)


def selectThresholdByCV(probs,gt):
    best_epsilon = 0
    best_f1 = 0
    f = 0
    stepsize = (max(probs) - min(probs)) / 1000;
    epsilons = np.arange(min(probs),max(probs),stepsize)
    for epsilon in np.nditer(epsilons):
        predictions = (probs < epsilon)
        f = f1_score(gt, predictions, average = "binary")
        if f > best_f1:
            best_f1 = f
            best_epsilon = epsilon
    return best_f1, best_epsilon


cpu_data = read_dataset('../datasets/dataset-cpu-training.csv')
network_data = read_dataset('../datasets/dataset-network-training.csv')
memory_data = read_dataset('../datasets/dataset-memory-training.csv')

# cpu_data = read_dataset('../datasets/dataset-cpu-validation.csv')
# network_data = read_dataset('../datasets/dataset-network-validation.csv')
# memory_data = read_dataset('../datasets/dataset-memory-validation.csv')

memory_data = memory_data[:4032]

del cpu_data['timestamp']
del network_data['timestamp']
del memory_data['timestamp']

data = [cpu_data, network_data, memory_data]
dataset = pd.concat(data, axis = 1)

n_training_samples = dataset.shape[0]

# Plot training dataset
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(dataset['cpu'], dataset['network'], dataset['memory'], c='b', marker='o')

# ax.set_xlabel('CPU usage')
# ax.set_ylabel('Network usage(MB/s)')
# ax.set_zlabel('Memory usage(MB)')

# plt.show()

# n_training_samples = tr_data.shape[0]
# n_dim = tr_data.shape[1]

# plt.figure()
# plt.xlabel("Latency (ms)")
# plt.ylabel("Throughput (mb/s)")
# plt.plot(tr_data[:,0],tr_data[:,1],"bx")
# plt.show()


mu, sigma = estimateGaussian(dataset)
print(mu)
print(sigma)
# p = multivariateGaussian(dataset,mu,sigma)

# outliers = np.asarray(np.where(p < ep))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

red = []
for x in range(0, n_training_samples):
    if dataset['memory'][x] > 1500000 or dataset['cpu'][x] > 55:
        red.append(True)
    else:
        red.append(False)


ax.scatter(dataset['cpu'], dataset['network'], dataset['memory'], c = ['r' if red[i] else 'b' for i in range(0, n_training_samples)], marker='o')
ax.set_xlabel('CPU usage')
ax.set_ylabel('Network usage(MB/s)')
ax.set_zlabel('Memory usage(MB)')

plt.show()

# plt.figure()
# plt.xlabel("Latency (ms)")
# plt.ylabel("Throughput (mb/s)")
# plt.plot(tr_data[:,0],tr_data[:,1],"bx") plt.plot(tr_data[outliers,0],tr_data[outliers,1],"ro")
# plt.show()
