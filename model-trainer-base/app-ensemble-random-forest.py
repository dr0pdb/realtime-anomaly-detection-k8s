import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from numpy import genfromtxt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from scipy.cluster.vq import kmeans
from sklearn.preprocessing import MinMaxScaler
from numpy.random import seed
import tensorflow
from keras.layers import Input, Dropout, Dense
from keras.models import Model, Sequential, load_model
from keras import regularizers
from keras.models import model_from_json
import seaborn as sns
sns.set(color_codes=True)


def read_dataset(filePath,delimiter=','):
    data = pd.read_csv(filePath) 
    return data


def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma


# define the autoencoder network model
def autoencoder_model(X):
    initializer = 'glorot_uniform'
    inputs = Input(shape=(X.shape[1],))
    L1 = Dense(12, activation='relu', kernel_initializer=initializer, 
               kernel_regularizer=regularizers.l2(0.00))(inputs)
    # L2 = Dense(12, activation='relu', kernel_initializer=initializer)(L1)
    output = Dense(X.shape[1], activation='relu', kernel_initializer=initializer)(L1)
    model = Model(inputs=inputs, outputs=output)
    return model


cpu_data = read_dataset('../datasets/dataset-cpu-training.csv')
network_data = read_dataset('../datasets/dataset-network-training.csv')
memory_data = read_dataset('../datasets/dataset-memory-training.csv')

# cpu_data = read_dataset('../datasets/dataset-cpu-validation.csv')
# network_data = read_dataset('../datasets/dataset-network-validation.csv')
# memory_data = read_dataset('../datasets/dataset-memory-validation.csv')

memory_data = memory_data[:8000]
cpu_data = cpu_data[:8000]
network_data = network_data[:8000]

del cpu_data['timestamp']
del network_data['timestamp']
del memory_data['timestamp']

data = [cpu_data, network_data, memory_data]
dataset = pd.concat(data, axis = 1)

n_training_samples = dataset.shape[0]

print("Loaded dataset")
print(dataset.head())

scaler = MinMaxScaler()
X_train = scaler.fit_transform(dataset)

seed(10)
tensorflow.random.set_seed(10)

model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mse')
model.summary()


# fit the model
nb_epochs = 5
batch_size = 5
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size, validation_split=0.05).history


# plot training losses
fig, ax = plt.subplots(figsize=(16,9))
ax.plot(history['loss'], 'b', label='Train', linewidth=3)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=3)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mse)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()

# plot the loss distribution on the training set
X_pred = model.predict(X_train)
X_pred = pd.DataFrame(X_pred, columns=dataset.columns)
X_pred.index = dataset.index

# scored = pd.DataFrame(index=dataset.index)
# scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis = 1)
# plt.figure(figsize=(16,9))
# plt.title('Loss Distribution', fontsize=16)
# sns.distplot(scored['Loss_mae'],
#              bins = 20, 
#              kde= True,
#             color = 'blue');
# plt.xlim([0.0,.5])
# plt.show()

# dataset['memory'].plot(kind='box')
# plt.show()



# mu, sigma = estimateGaussian(dataset)
# print(mu)
# print(sigma)
# # p = multivariateGaussian(dataset,mu,sigma)

# # outliers = np.asarray(np.where(p < ep))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# red = []
# for x in range(0, n_training_samples):
#     if dataset['memory'][x] > 1500000 or dataset['cpu'][x] > 55:
#         red.append(True)
#     else:
#         red.append(False)


# ax.scatter(dataset['cpu'], dataset['network'], dataset['memory'], c = ['r' if red[i] else 'b' for i in range(0, n_training_samples)], marker='o')
# ax.set_xlabel('CPU usage')
# ax.set_ylabel('Network usage(MB/s)')
# ax.set_zlabel('Memory usage(MB)')

# plt.show()

# plt.figure()
# plt.xlabel("Latency (ms)")
# plt.ylabel("Throughput (mb/s)")
# plt.plot(tr_data[:,0],tr_data[:,1],"bx") plt.plot(tr_data[outliers,0],tr_data[outliers,1],"ro")
# plt.show()
