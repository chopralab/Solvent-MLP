import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


y_val = []
y_val1 = np.load('data/k-fold-data/yval1.npy')
y_val2 = np.load('data/k-fold-data/yval2.npy')
y_val3 = np.load('data/k-fold-data/yval3.npy')
y_val4 = np.load('data/k-fold-data/yval4.npy')
y_val5 = np.load('data/k-fold-data/yval5.npy')
y_val = np.concatenate((y_val1,y_val2,y_val3,y_val4,y_val5))

y_gold = []
y_gold1 = np.load('data/k-fold-data/ygold1.npy')
y_gold2 = np.load('data/k-fold-data/ygold2.npy')
y_gold3 = np.load('data/k-fold-data/ygold3.npy')
y_gold4 = np.load('data/k-fold-data/ygold4.npy')
y_gold5 = np.load('data/k-fold-data/ygold5.npy')
y_gold = np.concatenate((y_gold1,y_gold2,y_gold3,y_gold4,y_gold5))

n_classes = 35
fpr = dict()
tpr = dict()
for i in range(n_classes):
	fpr[i], tpr[i], _ = roc_curve(y_val[:, i], y_gold[:, i])


fpr["micro"], tpr["micro"], _ = roc_curve(y_val.ravel(), y_gold.ravel())



plt.figure()
lw = 2
plt.xscale('log')
plt.plot(fpr[7], tpr[7], color='r',
         lw=lw, label='Carbon Tetrachloride')
plt.plot(fpr[11], tpr[11], color='cornflowerblue',
         lw=lw, label='DCM')
plt.plot(fpr[16], tpr[16], color='c',
         lw=lw, label='Dichloroethane')
plt.plot(fpr[9], tpr[9], color='m',
         lw=lw, label='Chloroform')
plt.plot(fpr[8], tpr[8], color='y',
         lw=lw, label='Chlorobenzene')
plt.plot(fpr[0], tpr[0], color='g',
         lw=lw, label='1,1-Dichloroethane')
plt.plot(np.arange(0, 1, 0.00001), np.arange(0, 1, 0.00001), color='navy', lw=lw, linestyle='--')
plt.xlim([0.00000125, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)
plt.title('Halogenated Solvents',fontsize=16)
plt.legend(loc="lower right")
out = 'Plotting/Plots/logROC-k1.pdf'
plt.savefig(out, dpi=600)
#plt.show()
plt.figure()


lw = 2
plt.xscale('log')
plt.plot(fpr[31], tpr[31], color='r',
         lw=lw, label='Toluene')
plt.plot(fpr[23], tpr[23], color='cornflowerblue',
         lw=lw, label='Hexane')
plt.plot(fpr[22], tpr[22], color='c',
         lw=lw, label='Heptane')
plt.plot(fpr[5], tpr[5], color='m',
         lw=lw, label='Benzene')
plt.plot(fpr[28], tpr[28], color='y',
         lw=lw, label='O-Xylene')
plt.plot(fpr[26], tpr[26], color='g',
         lw=lw, label='Methyl t-Butyl Ether')
plt.plot(fpr[10], tpr[10], color='darkorange',
         lw=lw, label='Cyclohexane')
plt.plot(np.arange(0, 1, 0.00001), np.arange(0, 1, 0.00001), color='navy', lw=lw, linestyle='--')
plt.xlim([0.00000125, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)
plt.title('Non-Polar Solvents',fontsize=16)
plt.legend(loc="lower right")
out = 'plotting/plots/logROC-k2.pdf'
plt.savefig(out, dpi=600)



plt.figure()
lw = 2
plt.xscale('log')
plt.plot(fpr[6], tpr[6], color='r',
         lw=lw, label='Butanone')
plt.plot(fpr[13], tpr[13], color='#ad8672',
         lw=lw, label='DMF')
plt.plot(fpr[12], tpr[12], color='c',
         lw=lw, label='DMAc')
plt.plot(fpr[3], tpr[3], color='m',
         lw=lw, label='Acetone')
plt.plot(fpr[30], tpr[30], color='y',
         lw=lw, label='THF')
plt.plot(fpr[15], tpr[15], color='g',
         lw=lw, label='DMSO')
plt.plot(fpr[27], tpr[27], color='darkorange',
         lw=lw, label='N-Methylpyrrolidone')
plt.plot(np.arange(0, 1, 0.00001), np.arange(0, 1, 0.00001), color='navy', lw=lw, linestyle='--')
plt.xlim([0.00000125, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)
plt.title('Polar-Aprotic Solvents',fontsize=16)
plt.legend(loc="lower right")
out = 'plotting/plots/logROC-k3.pdf'
plt.savefig(out, dpi=600)


plt.figure()
lw = 2
plt.xscale('log')
plt.plot(fpr[4], tpr[4], color='r',
         lw=lw, label='Acetonitrile')
plt.plot(fpr[1], tpr[1], color='cornflowerblue',
         lw=lw, label='1,4-Dioxane')
plt.plot(fpr[29], tpr[29], color='c',
         lw=lw, label='Pyridine')
plt.plot(fpr[21], tpr[21], color='m',
         lw=lw, label='Ethyl Acetate')
plt.plot(fpr[17], tpr[17], color='y',
         lw=lw, label='Diethyl Ether')
plt.plot(fpr[18], tpr[18], color='g',
         lw=lw, label='Diglyme')
plt.plot(fpr[19], tpr[19], color='darkorange',
         lw=lw, label='Dimethoxyethane')
plt.plot(fpr[14], tpr[14], color='hotpink',
         lw=lw, label='DMPU')
plt.plot(np.arange(0, 1, 0.00001), np.arange(0, 1, 0.00001), color='navy', lw=lw, linestyle='--')
plt.xlim([0.00000125, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)
plt.title('Polar-Aprotic Solvents Continued',fontsize=16)
plt.legend(loc="lower right")
out = 'plotting/plots/logROC-k4.pdf'
plt.savefig(out, dpi=600)


plt.figure()
lw = 2
plt.xscale('log')
plt.plot(fpr[33], tpr[33], color='r',
         lw=lw, label='n-Butanol')
plt.plot(fpr[25], tpr[25], color='cornflowerblue',
         lw=lw, label='Methanol')
plt.plot(fpr[20], tpr[20], color='c',
         lw=lw, label='Ethanol')
plt.plot(fpr[32], tpr[32], color='m',
         lw=lw, label='Water')
plt.plot(fpr[24], tpr[24], color='y',
         lw=lw, label='Isopropanol')
plt.plot(fpr[34], tpr[34], color='g',
         lw=lw, label='t-Butanol')
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='1-Propanol')
plt.plot(np.arange(0, 1, 0.00001), np.arange(0, 1, 0.00001), color='navy', lw=lw, linestyle='--')
plt.xlim([0.00000125, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)
plt.title('Polar-Protic Solvents',fontsize=16)
plt.legend(loc="lower right")
out = 'plotting/plots/logROC-k5.pdf'
plt.savefig(out, dpi=600)


###
plt.figure()
plt.xscale('log')
plt.plot(fpr[11], tpr[11], color='cornflowerblue',
         lw=lw, label='DCM')
plt.plot(fpr[30], tpr[30], color='y',
         lw=lw, label='THF')
plt.plot(fpr[32], tpr[32], color='m',
         lw=lw, label='Water')
plt.plot(fpr[31], tpr[31], color='r',
         lw=lw, label='Toluene')
plt.plot(fpr[13], tpr[13], color='#ad8672',
         lw=lw, label='DMF')
plt.plot(np.arange(0, 1, 0.00001), np.arange(0, 1, 0.00001), color='navy', lw=lw, linestyle='--')
plt.xlim([0.00000125, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)
plt.title('Receiver Operating Characteristic Curve',fontsize=16)
plt.legend(loc="lower right")
out = 'plotting/plots/mainROC.pdf'
plt.savefig(out, dpi=600)
###



