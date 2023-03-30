import numpy as np
from keras.models import model_from_json
import pandas as pd

dataframe = pd.read_csv("data/fingerprints/all_inhouse_bits.csv", header=None)
dataset = dataframe.values
X = dataset[:,1:].astype(int)

json_file = open('models/50-epoch-model/full-model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/50-epoch-model/full-model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


xd = loaded_model.predict( X, batch_size=None, verbose=0, steps=None)

np.save('data/inhouse-gold',np.array(xd))
