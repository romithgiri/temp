from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
dataset=loadtxt('diabetes.csv', delimiter=',')
dataset
