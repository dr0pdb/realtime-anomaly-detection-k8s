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
import math
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.tools as tls


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

my_order = (1, 1, 1)
my_seasonal_order = (0, 1, 1, 7)

history = [x for x in train_log]
predictions = list()
predict_log=list()
for t in range(len(test_log)):
    model = sm.tsa.SARIMAX(history, order=my_order, seasonal_order=my_seasonal_order,enforce_stationarity=False,enforce_invertibility=False)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    predict_log.append(output[0])
    yhat = 10**output[0]
    predictions.append(yhat)
    obs = test_log[t]
    history.append(obs)
   # print('predicted=%f, expected=%f' % (output[0], obs))
#error = math.sqrt(mean_squared_error(test_log, predict_log))
#print('Test rmse: %.3f' % error)
# plot
figsize=(12, 7)
plt.figure(figsize=figsize)
pyplot.plot(test,label='Actuals')
pyplot.plot(predictions, color='red',label='Predicted')
pyplot.legend(loc='upper right')
pyplot.show()