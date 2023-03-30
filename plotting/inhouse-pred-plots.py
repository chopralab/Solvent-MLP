import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import LabelEncoder
from copy import  deepcopy
import numpy as np
import pandas as pd
from statistics import harmonic_mean


def heatmap( data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="",  **kwargs):
    if not ax:
        ax = plt.gca()
    im = ax.imshow(data, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    plt.yticks( range(len(row_labels)) , row_labels , fontsize=6.8)
    plt.xticks( range(len(col_labels)) , col_labels ,rotation='vertical' , fontsize=6.8)
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


c = mcolors.ColorConverter().to_rgb
rvb = make_colormap(
    [ c('#ceecff'), c('#bfb4f1'), 0.6 , c('#bfb4f1'), c('#ffbb5f'), 0.9, c('#ffbb5f'), c('#ffc600'), 0.95, c('#ffc600'), c('#ff6757')])
N = 1000000000
array_dg = np.random.uniform(0, 10, size=(N, 2))
colors = np.random.uniform(-2, 2, size=(N,))


dataframe = pd.read_csv("data/fingerprints/all_inhouse_bits.csv", header=None)
dataset = dataframe.values
X = dataset[:,1:].astype(int)

Y =["DCM","THF","Methanol","Ethanol","DMF","Water","Acetonitrile","Toluene","Ethyl Acetate","1,4-Dioxane","Chloroform","Acetone","DMSO","Diethyl Ether","Isopropanol","Benzene","Carbon Tetrachloride","N-Methylpyrrolidone","Dichloroethane","n-Butanol","DMAc","Hexane","O-Xylene","Butanone","Pyridine","1,1-Dichloroethane","t-Butanol","Chlorobenzene","Methyl t-Butyl Ether","Dimethoxyethane","Heptane","Diglyme","Cyclohexane","1-Propanol","DMPU"]
Ys = np.sort(Y)
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(np.sort(Y))

dataframe = pd.read_csv("data/inhouseresults.csv", header=None)
dataset = dataframe.values
Y2 = dataset[:,0]


Yp = np.concatenate((Y,Y2))

encoder = LabelEncoder()
encoder.fit(Yp)
encoded_l = np.unique(encoder.transform(Y2))
encoded_tl = encoder.transform(Y)

xd = np.load('data/inhouse-gold.npy')




sorted_solv = [11, 30, 25, 20, 13, 32,  4, 31, 21,  1,  9,  3, 15, 17, 24,  5,  7, 27, 16, 33, 12, 23, 28,  6, 29,  0, 34,  8, 26, 19, 22, 18, 10,  2,14]

### does it normaly
xp = deepcopy(xd)
c = 0
f1=[]
### True label, guessed label , softmax value of argmax spot
while c < len(xp):
    vtg = []
    vtg.append(int(encoder.transform([Y2[c]])))
    vtg.append(int(encoder.transform([Y[sorted_solv.index(np.argmax(xp[c]))]])))
    vtg.append(xp[c][np.argmax(xp[c])])
    f1.append(vtg)
    print('{0}\r'.format(float(c)/len(xp))),
    c = c+1


spooky_array1 =[]
f1a= np.array(f1)
f1u = np.unique(f1a[:,0])#ground truth
f1u2 = np.unique(f1a[:,1])#predicted

for i in f1u:
    ts_array=[]
    for k in f1u2:
        t_array=[]
        g=0
        for j in f1:
            if j[0] == int(i):
                if j[1] == k:
                    g = g + 1
                    t_array.append(1/j[2])
        b = sum(t_array)
        if g != 0:
            b = g/b
        ts_array.append(b)
    spooky_array1.append(ts_array)

Y_l1 = []
for i in f1u:
	Y_l1.append(encoder.inverse_transform([int(i)]))

Y_l1 = np.concatenate((Y_l1))


X_l1 = []
for i in f1u2:
	X_l1.append(encoder.inverse_transform([int(i)]))

X_l1 = np.concatenate((X_l1))

print(X_l1)


fig , ax = plt.subplots()
im , cbar = heatmap( np.array(spooky_array1) , Y_l1, X_l1, ax=ax , cmap=rvb, cbarlabel="Softmax Value")
plt.xlabel('Predicted Solvent',fontsize=14)
plt.ylabel('Solvent Label',fontsize=14)
plt.title('Maximal Softmax Values',fontsize=16)
plt.tight_layout()
out = 'plotting/plots/fig-S6A.pdf'
plt.savefig(out,dpi=600)
plt.show()

####################################################

xp = deepcopy(xd)
c = 0
f1=[]
while c < len(xp):
	vtg = []
	qt = np.quantile(xp[c],0.95)
	g = 0
	if xp[c][np.argmax(xp[c])] <= 0.5:
		for i in xp[c]:
			vtg = []
			if i > qt:
				vtg.append(int(encoder.transform([Y2[c]])))
				vtg.append(int(encoder.transform([Y[sorted_solv.index(g)]])))
				vtg.append(i)
				f1.append(vtg)
			g = g + 1
	c += 1


spooky_array1 =[]
f1a= np.array(f1)
#f1u = np.unique(f1a[:,0])#ground truth
f1u2 = np.unique(f1a[:,1])#predicted

for i in f1u:
    ts_array=[]
    for k in f1u2:
        t_array=[]
        g=0
        for j in f1:
            if j[0] == int(i):
                if j[1] == k:
                    g = g + 1
                    t_array.append(1/j[2])
        b = sum(t_array)
        if g != 0:
            b = g/b
        ts_array.append(b)
    spooky_array1.append(ts_array)

Y_l1 = []
for i in f1u:
	Y_l1.append(encoder.inverse_transform([int(i)]))

Y_l1 = np.concatenate((Y_l1))


X_l1 = []
for i in f1u2:
	X_l1.append(encoder.inverse_transform([int(i)]))

X_l1 = np.concatenate((X_l1))

print(X_l1)


fig , ax = plt.subplots()
im , cbar = heatmap( np.array(spooky_array1) , Y_l1, X_l1, ax=ax , cmap=rvb, cbarlabel="Softmax Value")
plt.xlabel('Predicted Solvent',fontsize=14)
plt.ylabel('Solvent Label',fontsize=14)
plt.title('Multi-label Softmax Values\n( x < 0.5 )',fontsize=16)
plt.tight_layout()
out = 'plotting/plots/fig-S6B.pdf'
plt.savefig(out,dpi=600)
plt.show()

####################


xp = deepcopy(xd)
c = 0
f1=[]
while c < len(xp):
	vtg = []
	if xp[c][np.argmax(xp[c])] > 0.5:
		vtg.append(int(encoder.transform([Y2[c]])))
		vtg.append(int(encoder.transform([Y[sorted_solv.index(np.argmax(xp[c]))]])))
		vtg.append(xp[c][np.argmax(xp[c])])
		f1.append(vtg)
	c += 1        

spooky_array1 =[]
f1a= np.array(f1)
#f1u = np.unique(f1a[:,0])#ground truth
f1u2 = np.unique(f1a[:,1])#predicted

for i in f1u:
    ts_array=[]
    for k in f1u2:
        t_array=[]
        g=0
        for j in f1:
            if j[0] == int(i):
                if j[1] == k:
                    g = g + 1
                    t_array.append(1/j[2])
        b = sum(t_array)
        if g != 0:
            b = g/b
        ts_array.append(b)
    spooky_array1.append(ts_array)

Y_l1 = []
for i in f1u:
	Y_l1.append(encoder.inverse_transform([int(i)]))

Y_l1 = np.concatenate((Y_l1))


X_l1 = []
for i in f1u2:
	X_l1.append(encoder.inverse_transform([int(i)]))

X_l1 = np.concatenate((X_l1))

print(X_l1)


fig , ax = plt.subplots()
im , cbar = heatmap( np.array(spooky_array1) , Y_l1, X_l1, ax=ax , cmap=rvb, cbarlabel="Softmax Value")
plt.xlabel('Predicted Solvent',fontsize=14)
plt.ylabel('Solvent Label',fontsize=14)
plt.title('Maximal Softmax Values\n ( x > 0.5 )',fontsize=16)
plt.tight_layout()
out = 'plotting/plots/fig-S6C.pdf'
plt.savefig(out,dpi=600)
plt.show()

####################


xp = deepcopy(xd)
c = 0
f1=[]
while c < len(xp):
	vtg = []
	if xp[c][np.argmax(xp[c])] > 0.5:
		vtg.append(int(encoder.transform([Y2[c]])))
		vtg.append(int(encoder.transform([Y[sorted_solv.index(np.argmax(xp[c]))]])))
		vtg.append(xp[c][np.argmax(xp[c])])
		f1.append(vtg)
	c += 1        

c = 0
while c < len(xp):
	vtg = []
	qt = np.quantile(xp[c],0.95)
	g = 0
	if xp[c][np.argmax(xp[c])] <= 0.5:
		for i in xp[c]:
			vtg = []
			if i > qt:
				vtg.append(int(encoder.transform([Y2[c]])))
				vtg.append(int(encoder.transform([Y[sorted_solv.index(g)]])))
				vtg.append(i)
				f1.append(vtg)
			g = g + 1
	c += 1

spooky_array1 =[]
f1a= np.array(f1)
#f1u = np.unique(f1a[:,0])#ground truth
f1u2 = np.unique(f1a[:,1])#predicted

for i in f1u:
    ts_array=[]
    for k in f1u2:
        t_array=[]
        g=0
        for j in f1:
            if j[0] == int(i):
                if j[1] == k:
                    g = g + 1
                    t_array.append(1/j[2])
        b = sum(t_array)
        if g != 0:
            b = g/b
        ts_array.append(b)
    spooky_array1.append(ts_array)

Y_l1 = []
for i in f1u:
	Y_l1.append(encoder.inverse_transform([int(i)]))

Y_l1 = np.concatenate((Y_l1))


X_l1 = []
for i in f1u2:
	X_l1.append(encoder.inverse_transform([int(i)]))

X_l1 = np.concatenate((X_l1))

print(X_l1)


fig , ax = plt.subplots()
im , cbar = heatmap( np.array(spooky_array1) , Y_l1, X_l1, ax=ax , cmap=rvb, cbarlabel="Softmax Value")
plt.xlabel('Predicted Solvent',fontsize=14)
plt.ylabel('Solvent Label',fontsize=14)
plt.title('Multi-label Softmax Values',fontsize=16)
plt.tight_layout()
out = 'plotting/plots/fig-S6D.pdf'
plt.savefig(out,dpi=600)
plt.show()