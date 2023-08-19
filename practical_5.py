import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
#from keras.wrappers.scikit_learn import Keras Regressor
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
dataframe=pd.read_csv("Housing.csv", delim_whitespace=True, header=None)
dataset=dataframe.values
X=dataset [:, 0:13]
Y=dataset [:, :13]



def wider_model(my_param):
    model=Sequential()
    model.add(Dense (15,input_dim=13,kernel_initializer='normal', activation='relu'))
    model.add(Dense (13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1,kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp', KerasClassifier (model=wider_model, my_param=123)))
pipeline = Pipeline (estimators)
kfold=KFold(n_splits=10)
results=cross_val_score(pipeline,X,Y,cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))
