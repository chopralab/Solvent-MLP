import matplotlib
matplotlib.use('Agg')
import itertools
import pandas as pd 
import keras
import numpy
from keras.models import Sequential
import itertools
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.model_selection import KFold,GroupKFold
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
#from IPython.display import SVG,display
from keras.utils.vis_utils import model_to_dot
#import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import os
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


seed = 7
numpy.random.seed(seed)
dataframe = pd.read_csv("/storage/armen_beck/largedata1.csv", header=None)
dataset = dataframe.values
X = dataset[:,1:].astype(int)
Y = dataset[:,0]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
y = dummy_y


earlystopping=keras.callbacks.EarlyStopping(monitor='loss',
                              min_delta=0,
                              patience=5,
                              verbose=0, mode='auto')
 

def createmodel():
	model = Sequential()
	model.add(Dense(1330, activation='relu', input_dim=2048))
	model.add(Dropout(0.59949))
	model.add(Dense(725, activation='relu'))
	model.add(Dropout(0.14677))
	model.add(Dense(35, activation='softmax'))
	# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',
	              optimizer='adam')
	return model
 
print (np.sum(y,axis=1))
# ysum = y.sum(axis=0).astype(float)
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=4)
ls=[]
trainls=[]
conf=[]
conf2=[]
f1ls = []
f1tls = []
fpr = dict()
tpr = dict()
roc_auc = dict()
nvar = 1
y_valdata = []
val_predsdata = []
for train_index, val_index in kfold.split(X, encoded_Y):
	temp_ls = []
	model = createmodel()
	X_train, X_val = X[train_index], X[val_index]
	y_train, y_val = y[train_index], y[val_index]
	model.fit(X_train, y_train, validation_data=(X_val,y_val),callbacks=[earlystopping],epochs=100000, batch_size=500)
	preds = model.predict(X_val)
	val_preds = model.predict(X_val)
	preds = preds.argmax(axis=1)
	train = model.predict(X_train)
	train = train.argmax(axis=1)	
	#preds[preds>=0.5] = 1
	#preds[preds<0.5] = 0
	#for i,col in enumerate(Y.columns):
	#temp_ls.append()
	ls.append(f1_score(encoded_Y[val_index],preds,average='weighted'))
	trainls.append(f1_score(encoded_Y[train_index],train,average='weighted'))
	model_json = model.to_json()
	with open("/storage/armen_beck/med4paper/2med4papermodel"+str(nvar)+".json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights("/storage/armen_beck/med4paper/2med4papermodel"+str(nvar)+".h5")
	print("Saved model to disk")
	output_filename = '/storage/armen_beck/med4paper/2med4paperROCxvalfold'+str(nvar)
	np.save(output_filename, X_val)
	output_filename = '/storage/armen_beck/med4paper/2med4paperROCfoldyval'+str(nvar)
	np.save(output_filename, y_val)
	output_filename = '/storage/armen_beck/med4paper/2med4paperROCfoldygold'+str(nvar)
	np.save(output_filename, val_preds)
	nvar = nvar + 1

print (ls, trainls)





