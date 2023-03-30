import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import math
from scipy import stats

import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


solv_labels=["DCM","THF","Methanol","Ethanol","DMF","Water","Acetonitrile","Toluene","Ethyl Acetate","1,4-Dioxane","Chloroform","Acetone","DMSO","Diethyl Ether","Isopropanol","Benzene","Carbon Tetrachloride","N-Methylpyrrolidone","Dichloroethane","n-Butanol","DMAc","Hexane","O-Xylene","Butanone","Pyridine","1,1-Dichloroethane","t-Butanol","Chlorobenzene","Methyl t-Butyl Ether","Dimethoxyethane","Heptane","Diglyme","Cyclohexane","1-Propanol","DMPU"]
solv_count=[151122,130236,85078,80260,75180,49833,32972,32672,22242,21344,15210,14790,12708,8887,8808,8463,7905,4701,3802,3016,2547,2433,2226,2031,1971,1963,1527,994,874,664,584,505,441,431,63]


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
roc_auc2 = dict()
for i in range(n_classes):
	fpr[i], tpr[i], _ = roc_curve(y_val[:, i], y_gold[:, i])
	roc_auc2[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_val.ravel(), y_gold.ravel())
roc_auc2["micro"] = auc(fpr["micro"], tpr["micro"])

roc_auc = []
c = 0
while c < 35:
	roc_auc.append(np.log(roc_auc2[c])+1)
	c = c + 1

encoder = LabelEncoder()
encoder.fit(solv_labels)
encoded_Y = encoder.transform(solv_labels)

y_gold_en = []
for i in y_gold:
	y_gold_en.append(i.argmax())

y_val_en = []
for i in y_val:
	y_val_en.append(i.argmax())

f1=f1_score(y_val_en,y_gold_en,average=None)

sort_labels = []
for i in range(0,35):
	sort_labels.append(encoder.inverse_transform([i])[0])

print("sort labels")
print(sort_labels)

ordered_solv_count = []
for i in sort_labels:
	c = 0
	for b in solv_labels:
		if i == b:
			ordered_solv_count.append(math.log10(solv_count[c]))
		c = c + 1

aucval = []
for i in range(0,35):
	aucval.append(roc_auc2[i])

sauc = sorted(aucval, reverse = True)
sl = []
for i in sauc:
	c = 0
	for b in aucval:
		if i == b:
			sl.append(sort_labels[c])
		c = c + 1


#figure 3A
Halogenated = [ 0,2,19,21,25,26 ]
Nonpolar = [15,17,23,24,27,29,30]
PolarAprotic = [1,3,4,6,7,8,10,12,16,20,22,28,32,33,34]
PolarProtic = [5,9,11,13,14,18,31]
colors = []
for i in range(35):
	if i in Halogenated:
		colors.append('#8dd35f')
	elif i in Nonpolar:
		colors.append('#ffdd55')
	elif i in PolarAprotic:
		colors.append('#ff9955')
	elif i in PolarProtic:
		colors.append('#87aade')
	

plt.bar( range(0,35) , sauc ,color= colors)
plt.ylim([.96,1])
plt.xlim([-0.6,34.5])
plt.xticks( range(0,35) , sl , rotation = '90')
plt.ylabel("ROC AUC's",fontsize=14)
plt.title("Area Under the Curve Values",fontsize=16)
plt.tight_layout()
out = 'plotting/plots/fig-3A.pdf'
plt.savefig(out,dpi=600)
plt.show()

sf1 = sorted(f1, reverse = True)
sl = []
for i in sf1:
	c = 0
	for b in f1:
		if i == b:
			sl.append(sort_labels[c])
		c = c + 1


Halogenated = [ 0,1,7,14,21,29 ]
Nonpolar = [18,20,25,26,28,31,33]
PolarAprotic = [2,3,4,5,6,8,11,12,15,16,22,24,27,30,32]
PolarProtic = [9,10,13,17,19,23,34]
colors = []
for i in range(35):
	if i in Halogenated:
		colors.append('#8dd35f')
	elif i in Nonpolar:
		colors.append('#ffdd55')
	elif i in PolarAprotic:
		colors.append('#ff9955')
	elif i in PolarProtic:
		colors.append('#87aade')

plt.bar( range(0,35) , sf1 , color=colors)
plt.plot((-2,37),(0.94384392,0.94384392),'--',linewidth=1,color='black',label='Average F1-Score Across All Solvents')
plt.ylim([.75,1])
plt.xlim([-0.6,34.5])
plt.xticks( range(0,35) , sl , rotation = '90')
plt.ylabel("F1-Score",fontsize=14)
plt.title("Aggregate F1-Scores Through K-Fold Validation",fontsize=16)
plt.legend(loc="upper right")
plt.tight_layout()
out = 'plotting/plots/fig-3C.pdf'
plt.savefig(out,dpi=600)
plt.show()


#Figure 3D
r2 = []
plt.figure()
plt.ylim([0.80, 1.0])
plt.xlim([math.log10(50), math.log10(170000)])
x = [ ordered_solv_count[0],ordered_solv_count[7],ordered_solv_count[8],ordered_solv_count[9],ordered_solv_count[11],ordered_solv_count[16]] 
y = [ f1[0],f1[7],f1[8],f1[9],f1[11],f1[16] ] 
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
#plt.xscale('log')
plt.scatter(x,y,color='#8dd35f',marker='^',label='Halogenated')
xl = range(0,150000)
yl = xl*slope+intercept
plt.plot( xl,yl,color='#8dd35f')
r2.append(r_value**2)

x = [ ordered_solv_count[31],ordered_solv_count[22],ordered_solv_count[23],ordered_solv_count[26],ordered_solv_count[28],ordered_solv_count[10],ordered_solv_count[5]] 
y = [ f1[31],f1[22],f1[23],f1[26],f1[28],f1[10],f1[5]] 
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
yl = xl*slope+intercept
plt.scatter(x,y,color='#ffdd55', marker='s', label='Nonpolar')
plt.plot( xl,yl,color='#ffdd55')
r2.append(r_value**2)


x = [ ordered_solv_count[1],ordered_solv_count[3],ordered_solv_count[4],ordered_solv_count[6],ordered_solv_count[12],ordered_solv_count[13],ordered_solv_count[14],ordered_solv_count[15],ordered_solv_count[17],ordered_solv_count[18],ordered_solv_count[19],ordered_solv_count[21],ordered_solv_count[27],ordered_solv_count[29],ordered_solv_count[30]]
y = [ f1[1],f1[3],f1[4],f1[6],f1[12],f1[13],f1[14],f1[15],f1[17],f1[18],f1[19],f1[21],f1[27],f1[29],f1[30]]
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
yl = xl*slope+intercept
plt.scatter(x,y,color='#ff9955', marker='o', label = 'Polar-Aprotic')
plt.plot( xl,yl,color='#ff9955')
r2.append(r_value**2)

x = [ ordered_solv_count[32],ordered_solv_count[25],ordered_solv_count[20],ordered_solv_count[24],ordered_solv_count[2],ordered_solv_count[33],ordered_solv_count[34] ] 
y = [ f1[32],f1[25],f1[20],f1[24],f1[2],f1[33],f1[34] ] 
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
yl = xl*slope+intercept
plt.scatter(x,y,color='#87aade', marker='d', label = 'Polar-Protic')
plt.plot( xl,yl,color='#87aade')
r2.append(r_value**2)

x,y = ordered_solv_count, f1
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
yl = xl*slope+intercept
plt.plot( xl,yl,color='k', linestyle='--')

plt.xlabel('Label Count (log)',fontsize=14)
plt.ylabel('F1-Score',fontsize=14)
plt.title('F1-Scores For Solvents vs Frequency in Training Set',fontsize=16)
plt.legend(loc="lower right")
out = 'plotting/plots/fig-3D.pdf'
plt.savefig(out,dpi=600)
plt.show()


y_gold_en = []
for i in y_gold1:
	y_gold_en.append(i.argmax())

y_val_en = []
for i in y_val1:
	y_val_en.append(i.argmax())

f1=f1_score(y_val_en,y_gold_en,average='weighted')