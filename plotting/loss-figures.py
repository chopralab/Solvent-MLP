[[1124, 0.52846, 486, 0.07836, 0.8511238348943962],[849, 0.43921, 399, 0.15228, 0.8432445658343699],[1306, 0.58109, 573, 0.0203, 0.8538169182924481],[1125, 0.54475, 646, 0.20657, 0.852485570603232],[1135, 0.4566, 603, 0.29252, 0.849126845484237],[1244, 0.38722, 542, 0.35424, 0.8489028851729601],[1012, 0.47514, 685, 0.36123, 0.8460537846267547],[896, 0.45826, 593, 0.34077, 0.843302689268158],[1060, 0.43077, 341, 0.00565, 0.8442606384711698],[1096, 0.35729, 432, 0.15312, 0.8474738616662463],[1178, 0.37433, 700, 0.11059, 0.8495251130999733],[1256, 0.42835, 435, 0.03514, 0.8491378717102153],[1021, 0.34671, 615, 0.11129, 0.8458515672687449],[1025, 0.43, 683, 0.01971, 0.8477719224719299],[1050, 0.48014, 822, 0.11561, 0.8505610552901238],[968, 0.42981, 734, 0.09514, 0.8458619921276936],[1071, 0.47654, 418, 0.20585, 0.8471630759205969],[1105, 0.44272, 421, 0.31965, 0.8473572042733487],[803, 0.4937, 442, 0.17943, 0.8420771830746667],[1128, 0.50395, 447, 0.18244, 0.8496643156516942],[1237, 0.5127, 193, 0.1497, 0.8474388559152862],[1141, 0.60046, 234, 0.04326, 0.8453369455462191],[1151, 0.51399, 195, 0.12415, 0.8473303490805777],[1249, 0.54778, 155, 0.05471, 0.8483692320508079],[1296, 0.53111, 317, 0.32947, 0.8516165188968831],[1260, 0.40285, 387, 0.29664, 0.8504658410100054],[1309, 0.52, 506, 0.4067, 0.850284638896068],[1302, 0.47917, 271, 0.36151, 0.8490543831195115],[1014, 0.29232, 521, 0.24861, 0.8434267465316526],[1137, 0.31743, 483, 0.21388, 0.84603698573415],[1260, 0.39174, 226, 0.18428, 0.8477884335938947],[814, 0.39208, 466, 0.23807, 0.8394909549046555]]

import matplotlib.pyplot as plt
import pandas as pd

import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

data209 = pd.read_csv("data/loss-data/loss-209.csv", header=None).values
l209 = data209[:,1]
al = pd.read_csv("data/loss-data/kfold-ave-losses.csv", header=None).values


kfoldvalloss = pd.read_csv("data/loss-data/kfoldvalloss4paper2.csv", header=None).values
kfoldloss = pd.read_csv("data/loss-data/kfoldloss4paper2.csv", header=None).values


plt.plot(al[:,0], color='b', linestyle='-', label='Validation Loss')
plt.plot(al[:,1], color='b', linestyle='--', label='Training Loss')
plt.xlabel('Epoch',fontsize=18)
plt.ylabel('Loss',fontsize=18)
plt.title('Average Loss Per Epoch Through K-Fold Validation',fontsize=16)
plt.legend(loc="upper right")
plt.tight_layout()
out = 'plotting/plots/loss-4B.pdf'
plt.savefig(out, dpi=600)
plt.show()

lw = 2
plt.figure()
plt.plot( kfoldvalloss[180:,1], color='r',
         lw=lw, label='Fold 5 validation loss')
plt.plot( kfoldloss[180:,1], color='b',
         lw=lw, label='Fold 5 training loss', linestyle='--')
plt.xlabel('Epoch',fontsize=14)
plt.ylabel('Loss',fontsize=14)
plt.title('Loss Per Epoch Fold 5 of 5',fontsize=16)
out = 'plotting/plots/k5.pdf'
plt.savefig(out, dpi=600)
plt.show()

plt.figure()
plt.plot( kfoldvalloss[130:179,1], color='r',
         lw=lw, label='Fold 4 validation loss')
plt.plot( kfoldloss[130:179,1], color='b',
         lw=lw, label='Fold 4 training loss', linestyle='--')
plt.xlabel('Epoch',fontsize=14)
plt.ylabel('Loss',fontsize=14)
plt.title('Loss Per Epoch Fold 4 of 5',fontsize=16)
out = 'plotting/plots/k4.pdf'
plt.savefig(out, dpi=600)
plt.show()


lw = 2
plt.figure()
plt.plot( kfoldvalloss[78:129,1], color='r',
         lw=lw, label='Fold 3 validation loss')
plt.plot(kfoldloss[78:129,1], color='b',
         lw=lw, label='Fold 3 training loss', linestyle='--')
plt.xlabel('Epoch',fontsize=14)
plt.ylabel('Loss',fontsize=14)
plt.title('Loss Per Epoch Fold 3 of 5',fontsize=16)
out = 'plotting/plots/k3.pdf'
plt.savefig(out, dpi=600)
plt.show()

plt.figure()
plt.plot( kfoldvalloss[39:77,1], color='r',
         lw=lw, label='Fold 2 validation loss')
plt.plot( kfoldloss[39:77,1], color='b',
         lw=lw, label='Fold 2 training loss', linestyle='--')
plt.xlabel('Epoch',fontsize=14)
plt.ylabel('Loss',fontsize=14)
plt.title('Loss Per Epoch Fold 2 of 5',fontsize=16)
out = 'plotting/plots/k2.pdf'
plt.savefig(out, dpi=600)
plt.show()

plt.figure()
plt.plot(range(0,38), kfoldvalloss[0:38,1], color='r',
         lw=lw, label='Fold 1 validation loss')
plt.plot(range(0,38), kfoldloss[0:38,1], color='b',
         lw=lw, label='Fold 1 training loss', linestyle='--')
plt.xlabel('Epoch',fontsize=14)
plt.ylabel('Loss',fontsize=14)
plt.title('Loss Per Epoch Fold 1 of 5',fontsize=16)
out = 'plotting/plots/k1.pdf'
plt.savefig(out, dpi=600)
plt.show()
lw = 2



###figure 3a###

valloss2 =   kfoldvalloss[39:79,1]
#train loss
plt.figure()
lw = 2
plt.plot( l209, color='b',
         lw=lw, label='Loss Monitored Model')

plt.plot( [0],[0], color='r',
         lw=lw, label='Validation Loss Tangent', linestyle='-')
plt.plot( [0],[0], color='b',
         lw=lw, label='Training Loss Tangent', linestyle='--')
plt.plot( [-500],[0], color='0.5',
         lw=lw, label='Loss Value at Stop', marker='o')
plt.plot( [28,48],[0.2663,0.2603], color='r',
         lw=.8, linestyle='-')
plt.plot( [29,49],[0.2561,0.2621], color='r',
         lw=.8,  linestyle='-')
plt.plot( [42,62],[0.2758,0.2378], color='r',
         lw=.8, linestyle='-')
plt.plot( [40,60],[0.2542,0.2542], color='r',
         lw=.8, linestyle='-')
plt.plot( [28,48],[0.2733,0.2613], color='r',
         lw=.8, linestyle='-')
plt.plot( [28,48],[0.2127,0.1487], color='b',
         lw=.8, linestyle='--')
plt.plot( [29,49],[0.2236,0.1296], color='b',
         lw=.8, linestyle='--')
plt.plot( [42,62],[0.1478,0.1578], color='b',
         lw=lw, linestyle='--')
plt.plot( [40,60],[0.1653,0.1473], color='b',
         lw=lw, linestyle='--')
plt.plot( [28,48],[0.2062,0.1542], color='b',
         lw=lw, linestyle='--')

plt.plot( [30,60],[0.14,0.14], color='.1',
         lw=1.5, linestyle='-')
plt.plot( [30,60],[0.28,0.28], color='0.1',
         lw=1.5, linestyle='-')
plt.plot( [30,30],[0.14,0.28], color='.1',
         lw=1.5, linestyle='-')
plt.plot( [60,60],[0.14,0.28], color='0.1',
         lw=1.5, linestyle='-')

plt.plot( [60,80],[0.28,.4], color='0.1',
         lw=1.5, linestyle='-')
plt.plot( [38],[0.2633], marker = 'o' , color ='0.1' ,markersize=4)
plt.plot( [39],[0.2591], marker = 'o', color = '0.1' ,markersize=4)
plt.plot( [52],[0.2568], marker = 'o' ,color = '0.1',markersize=4)
plt.plot( [50],[0.2542], marker = 'o' , color ='0.1',markersize=4)
plt.plot( [38],[0.2673], marker = 'o', color = '0.1',markersize=4)
plt.plot( [38],[0.1807], marker = 'o' , color ='0.1',markersize=4)
plt.plot( [39],[0.1766], marker = 'o', color = '0.1',markersize=4)
plt.plot( [52],[0.1528], marker = 'o', color = '0.1',markersize=4)
plt.plot( [50],[0.1563], marker = 'o', color = '0.1',markersize=4)
plt.plot( [38],[0.1802], marker = 'o', color = '0.1',markersize=4)
plt.xlabel('Epoch',fontsize=14)
plt.ylabel('Loss',fontsize=14)
plt.xlim([0.0, 210])
plt.title('Loss Per Epoch',fontsize=16)
plt.legend(loc="upper left")
out = 'plotting/plots/loss-3a.svg'
plt.savefig(out, dpi=600)
plt.show()

##figure 3a inset###
plt.plot( [50,50],[0,1], color='black', lw=.5)
plt.plot( l209, color='grey',
        lw=lw)

plt.plot( [28,48],[0.2663,0.2603], color='r',
         lw=lw, linestyle='-')
plt.plot( [29,49],[0.2561,0.2621], color='r',
         lw=lw, linestyle='-')
plt.plot( [42,62],[0.2758,0.2378], color='r',
         lw=lw, linestyle='-')
plt.plot( [40,60],[0.2542,0.2542], color='r',
         lw=lw, linestyle='-')
plt.plot( [28,48],[0.2733,0.2613], color='r',
         lw=lw, linestyle='-')
plt.plot( [28,48],[0.2127,0.1487], color='b',
         lw=lw, linestyle='--')
plt.plot( [29,49],[0.2236,0.1296], color='b',
         lw=lw, linestyle='--')
plt.plot( [42,62],[0.1478,0.1578], color='b',
         lw=lw, linestyle='--')
plt.plot( [40,60],[0.1653,0.1473], color='b',
         lw=lw, linestyle='--')
plt.plot( [28,48],[0.2062,0.1542], color='b',
         lw=lw, linestyle='--')
plt.plot( [38],[0.2633], marker = 'o' , color ='0.1')
plt.plot( [39],[0.2591], marker = 'o', color = '0.1')
plt.plot( [52],[0.2568], marker = 'o' ,color = '0.1')
plt.plot( [50],[0.2542], marker = 'o' , color ='0.1')
plt.plot( [38],[0.2673], marker = 'o', color = '0.1')
plt.plot( [38],[0.1807], marker = 'o' , color ='0.1')
plt.plot( [39],[0.1766], marker = 'o', color = '0.1')
plt.plot( [52],[0.1528], marker = 'o', color = '0.1')
plt.plot( [50],[0.1563], marker = 'o', color = '0.1')
plt.plot( [38],[0.1802], marker = 'o', color = '0.1')
plt.xlim([30.0, 60.0])
plt.ylim([0.14, .27])
out = 'plotting/plots/loss-3ai.pdf'
plt.savefig(out, dpi=600)
plt.show()
