import matplotlib.pyplot as plt
#import pandas as pd 
import matplotlib.colors as mcolors
from sklearn.preprocessing import LabelEncoder
from copy import  deepcopy
import numpy as np
import pandas as pd
from statistics import harmonic_mean

Y =["DCM","THF","Methanol","Ethanol","DMF","Water","Acetonitrile","Toluene","Ethyl Acetate","1,4-Dioxane","Chloroform","Acetone","DMSO","Diethyl Ether","Isopropanol","Benzene","Carbon Tetrachloride","N-Methylpyrrolidone","Dichloroethane","n-Butanol","DMAc","Hexane","O-Xylene","Butanone","Pyridine","1,1-Dichloroethane","t-Butanol","Chlorobenzene","Methyl t-Butyl Ether","Dimethoxyethane","Heptane","Diglyme","Cyclohexane","1-Propanol","DMPU"]
solv_labels=["DCM","THF","Methanol","Ethanol","DMF","Water","Acetonitrile","Toluene","Ethyl Acetate","1,4-Dioxane","Chloroform","Acetone","DMSO","Diethyl Ether","Isopropanol","Benzene","Carbon Tetrachloride","N-Methylpyrrolidone","Dichloroethane","n-Butanol","DMAc","Hexane","O-Xylene","Butanone","Pyridine","1,1-Dichloroethane","t-Butanol","Chlorobenzene","Methyl t-Butyl Ether","Dimethoxyethane","Heptane","Diglyme","Cyclohexane","1-Propanol","DMPU"]
sl = sorted(solv_labels)
Y = solv_labels

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
sl = sorted( solv_labels )

sorted_solv = [11, 30, 25, 20, 13, 32,  4, 31, 21,  1,  9,  3, 15, 17, 24,  5,  7, 27, 16, 33, 12, 23, 28,  6, 29,  0, 34,  8, 26, 19, 22, 18, 10,  2,14]


y_gold = []
y_gold1 = np.load('data/k-fold-data/ygold1.npy')
y_gold2 = np.load('data/k-fold-data/ygold2.npy')
y_gold3 = np.load('data/k-fold-data/ygold3.npy')
y_gold4 = np.load('data/k-fold-data/ygold4.npy')
y_gold5 = np.load('data/k-fold-data/ygold5.npy')
y_gold = np.concatenate((y_gold1,y_gold2,y_gold3,y_gold4,y_gold5))

y_val = []
y_val1 = np.load('data/k-fold-data/yval1.npy')
y_val2 = np.load('data/k-fold-data/yval2.npy')
y_val3 = np.load('data/k-fold-data/yval3.npy')
y_val4 = np.load('data/k-fold-data/yval4.npy')
y_val5 = np.load('data/k-fold-data/yval5.npy')
y_val = np.concatenate((y_val1,y_val2,y_val3,y_val4,y_val5))


def heatmap( data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="",  **kwargs):
    if not ax:
        ax = plt.gca()
    im = ax.imshow(data, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    plt.yticks( range(len(solv_labelsx)) , row_labels , fontsize=6.8)
    plt.xticks( range(len(solv_labelsy)) , col_labels ,rotation='vertical' , fontsize=6.8)
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

solv_labelsx = Y
solv_labelsy = Y


############### Maximal SoftMax ##############

Y2 = []
for rxn in y_val:
    Y2.append(int(np.argmax(rxn)))


xp = deepcopy(y_gold)
c = 0
f1=[]
### True label, guessed label , softmax value of argmax spot
while c < len(xp):
    vtg = []
    vtg.append(Y2[c])
    vtg.append(int(np.argmax(xp[c])))
    vtg.append(np.max(xp[c]))
    f1.append(vtg)
    c = c+1

hm_matrix = []
for i in range(35):
    hm_matrix.append([])
    for j in range(35):
        hm_matrix[-1].append([])

for vals in f1:
    hm_matrix[sorted_solv.index(vals[0])][sorted_solv.index(vals[1])].append(vals[2])

for i, row in enumerate(hm_matrix):
    for j, vals in enumerate(row):
        if vals != []:
            hm_matrix[i][j] = harmonic_mean(vals)
        else:
            hm_matrix[i][j] = 0


fig , ax = plt.subplots()
im , cbar = heatmap( np.array(hm_matrix) , Y, Y, ax=ax , cmap=rvb, cbarlabel="Softmax Value")
plt.xlabel('Predicted Solvent',fontsize=14)
plt.ylabel('Solvent Label',fontsize=14)
plt.title('Maximal Softmax Values',fontsize=16)
plt.tight_layout()
out = 'plotting/plots/fig-S5A.pdf'
plt.savefig(out,dpi=600)
plt.show()

############### AUC Ordering ################# 
order = [7,11,16,9,8,0,31,23,22,5,28,26,10,6,13,12,3,30,15,27,4,1,29,21,17,18,19,14,33,25,20,32,24,34,2]

new_labels = []
for i in order:
	new_labels.append(sl[i])

#new matrix
auc_hm_matrix = []     
for z in range(35):
    auc_hm_matrix.append([])
    for y in range(35):
        auc_hm_matrix[-1].append([])

#reorder matrix based on sorted_solvent (hm_matrix)     
for i, row in enumerate(order):
    for j, vals in enumerate(order):
            auc_hm_matrix[i][j] = hm_matrix[sorted_solv.index(row)][sorted_solv.index(vals)]

fig , ax = plt.subplots()
im , cbar = heatmap( np.array(auc_hm_matrix) , new_labels, new_labels, ax=ax , cmap=rvb, cbarlabel="Softmax Value")
plt.xlabel('Predicted Solvent',fontsize=14)
plt.ylabel('Solvent Label',fontsize=14)
plt.title('Maximal Softmax Values',fontsize=16)
plt.tight_layout()
out = 'plotting/plots/fig-S4A.pdf'
plt.savefig(out,dpi=600)
plt.show()
#################################################

################ Only multilabel y_gold < 0.5 ##########

#this is being repeated because I want every code block to be self sustained
Y2 = []
for rxn in y_val:
    Y2.append(int(np.argmax(rxn)))


xp = deepcopy(y_gold)
c = 0
f2 = []
while c < len(xp):
	vtg = []
	qt = np.quantile(xp[c],0.95)
	g = 0
	if xp[c][np.argmax(xp[c])] <= 0.5:
		for i in xp[c]:
			vtg = []
			if i > qt:
				vtg.append(Y2[c])
				vtg.append(g)
				vtg.append(i)
				f2.append(vtg)
			g = g + 1
	c += 1
        

hm_matrix = []
for i in range(35):
    hm_matrix.append([])
    for j in range(35):
        hm_matrix[-1].append([])

for vals in f2:
    hm_matrix[sorted_solv.index(vals[0])][sorted_solv.index(vals[1])].append(vals[2])

for i, row in enumerate(hm_matrix):
    for j, vals in enumerate(row):
        if vals != []:
            hm_matrix[i][j] = harmonic_mean(vals)
        else:
            hm_matrix[i][j] = 0


fig , ax = plt.subplots()
im , cbar = heatmap( np.array(hm_matrix) , Y, Y, ax=ax , cmap=rvb, cbarlabel="Softmax Value")
plt.xlabel('Predicted Solvent',fontsize=14)
plt.ylabel('Solvent Label',fontsize=14)
plt.title('Multi-label Softmax Values\n x < 0.5',fontsize=16)
plt.tight_layout()
out = 'plotting/plots/fig-S5B.pdf'
plt.savefig(out,dpi=600)
plt.show()

############### AUC Ordering ################# 

order = [7,11,16,9,8,0,31,23,22,5,28,26,10,6,13,12,3,30,15,27,4,1,29,21,17,18,19,14,33,25,20,32,24,34,2]

new_labels = []
for i in order:
	new_labels.append(sl[i])

#new matrix
auc_hm_matrix = []     
for z in range(35):
    auc_hm_matrix.append([])
    for y in range(35):
        auc_hm_matrix[-1].append([])

#reorder matrix based on sorted_solvent (hm_matrix)     
for i, row in enumerate(order):
    for j, vals in enumerate(order):
            auc_hm_matrix[i][j] = hm_matrix[sorted_solv.index(row)][sorted_solv.index(vals)]

fig , ax = plt.subplots()
im , cbar = heatmap( np.array(auc_hm_matrix) , new_labels, new_labels, ax=ax , cmap=rvb, cbarlabel="Softmax Value")
plt.xlabel('Predicted Solvent',fontsize=14)
plt.ylabel('Solvent Label',fontsize=14)
plt.title('Multi-label Softmax Values\n x < 0.5',fontsize=16)
plt.tight_layout()
out = 'plotting/plots/fig-S4B.pdf'
plt.savefig(out,dpi=600)
plt.show()

#################################################

################ Only  y_gold > 0.5 ##########


#this is being repeated because I want every code block to be self sustained
Y2 = []
for rxn in y_val:
    Y2.append(int(np.argmax(rxn)))


xp = deepcopy(y_gold)
c = 0
f2 = []
while c < len(xp):
	vtg = []
	if xp[c][np.argmax(xp[c])] > 0.5:
		vtg.append(Y2[c])
		vtg.append(np.argmax(xp[c]))
		vtg.append(xp[c][np.argmax(xp[c])])
		f2.append(vtg)
	c += 1
        
hm_matrix = []
for i in range(35):
    hm_matrix.append([])
    for j in range(35):
        hm_matrix[-1].append([])

for vals in f2:
    hm_matrix[sorted_solv.index(vals[0])][sorted_solv.index(vals[1])].append(vals[2])

for i, row in enumerate(hm_matrix):
    for j, vals in enumerate(row):
        if vals != []:
            hm_matrix[i][j] = harmonic_mean(vals)
        else:
            hm_matrix[i][j] = 0


fig , ax = plt.subplots()
im , cbar = heatmap( np.array(hm_matrix) , Y, Y, ax=ax , cmap=rvb, cbarlabel="Softmax Value")
plt.xlabel('Predicted Solvent',fontsize=14)
plt.ylabel('Solvent Label',fontsize=14)
plt.title('Maximal Softmax Values\n x > 0.5',fontsize=16)
plt.tight_layout()
out = 'plotting/plots/fig-S5C.pdf'
plt.savefig(out,dpi=600)
plt.show()

############### AUC Ordering ################# 

order = [7,11,16,9,8,0,31,23,22,5,28,26,10,6,13,12,3,30,15,27,4,1,29,21,17,18,19,14,33,25,20,32,24,34,2]

new_labels = []
for i in order:
	new_labels.append(sl[i])

#new matrix
auc_hm_matrix = []     
for z in range(35):
    auc_hm_matrix.append([])
    for y in range(35):
        auc_hm_matrix[-1].append([])

#reorder matrix based on sorted_solvent (hm_matrix)     
for i, row in enumerate(order):
    for j, vals in enumerate(order):
            auc_hm_matrix[i][j] = hm_matrix[sorted_solv.index(row)][sorted_solv.index(vals)]

fig , ax = plt.subplots()
im , cbar = heatmap( np.array(auc_hm_matrix) , new_labels, new_labels, ax=ax , cmap=rvb, cbarlabel="Softmax Value")
plt.xlabel('Predicted Solvent',fontsize=14)
plt.ylabel('Solvent Label',fontsize=14)
plt.title('Multi-label Softmax Values\n x > 0.5',fontsize=16)
plt.tight_layout()
out = 'plotting/plots/fig-S4C.pdf'
plt.savefig(out,dpi=600)
plt.show()


#################################################

################ Multi-label ####################

#this is being repeated because I want every code block to be self sustained
Y2 = []
for rxn in y_val:
    Y2.append(int(np.argmax(rxn)))


xp = deepcopy(y_gold)
c = 0
f2 = []
while c < len(xp):
	vtg = []
	if xp[c][np.argmax(xp[c])] > 0.5:
		vtg.append(Y2[c])
		vtg.append(np.argmax(xp[c]))
		vtg.append(xp[c][np.argmax(xp[c])])
		f2.append(vtg)
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
				vtg.append(Y2[c])
				vtg.append(g)
				vtg.append(i)
				f2.append(vtg)
			g = g + 1
	c += 1        

        
hm_matrix = []
for i in range(35):
    hm_matrix.append([])
    for j in range(35):
        hm_matrix[-1].append([])

for vals in f2:
    hm_matrix[sorted_solv.index(vals[0])][sorted_solv.index(vals[1])].append(vals[2])

for i, row in enumerate(hm_matrix):
    for j, vals in enumerate(row):
        if vals != []:
            hm_matrix[i][j] = harmonic_mean(vals)
        else:
            hm_matrix[i][j] = 0


fig , ax = plt.subplots()
im , cbar = heatmap( np.array(hm_matrix) , Y, Y, ax=ax , cmap=rvb, cbarlabel="Softmax Value")
plt.xlabel('Predicted Solvent',fontsize=14)
plt.ylabel('Solvent Label',fontsize=14)
plt.title('Multi-label Softmax Values',fontsize=16)
plt.tight_layout()
out = 'plotting/plots/fig-S5D.pdf'
plt.savefig(out,dpi=600)
plt.show()

############### AUC Ordering ################# 

order = [7,11,16,9,8,0,31,23,22,5,28,26,10,6,13,12,3,30,15,27,4,1,29,21,17,18,19,14,33,25,20,32,24,34,2]

new_labels = []
for i in order:
	new_labels.append(sl[i])

#new matrix
auc_hm_matrix = []     
for z in range(35):
    auc_hm_matrix.append([])
    for y in range(35):
        auc_hm_matrix[-1].append([])

#reorder matrix based on sorted_solvent (hm_matrix)     
for i, row in enumerate(order):
    for j, vals in enumerate(order):
            auc_hm_matrix[i][j] = hm_matrix[sorted_solv.index(row)][sorted_solv.index(vals)]

fig , ax = plt.subplots()
im , cbar = heatmap( np.array(auc_hm_matrix) , new_labels, new_labels, ax=ax , cmap=rvb, cbarlabel="Softmax Value")
plt.xlabel('Predicted Solvent',fontsize=14)
plt.ylabel('Solvent Label',fontsize=14)
plt.title('Multi-label Softmax Values',fontsize=16)
plt.tight_layout()
out = 'plotting/plots/fig-S4D.pdf'
plt.savefig(out,dpi=600)
plt.show()