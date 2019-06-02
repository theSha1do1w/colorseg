import scipy.io
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

mat = scipy.io.loadmat('./data/mturkData.mat')
source_x = mat['data'].reshape(10743, 15)
source_y = mat['targets'].reshape(10743,)

train_X, test_X, train_y, test_y = train_test_split(source_x, source_y, test_size=.1)

forests = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=-1)
forests.fit(train_X, train_y)


def pred(color):
    rate = forests.predict(color.reshape(1, 15)/255)
    return rate[0]




