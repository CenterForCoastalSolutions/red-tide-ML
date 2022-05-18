import numpy as np
import matplotlib.pyplot as plt
import math
from os import listdir
from os.path import isfile, join

configfilename = 'date_train_test_depth_norm_w_knn_50k_hill'

random_files = ['roc_curve_info/random_train_test_depth_norm_w_knn.npy',\
				'roc_curve_info/random_Stumpf.npy',\
				'roc_curve_info/random_Shehhi.npy',\
				'roc_curve_info/random_Soto.npy',\
				'roc_curve_info/random_Amin.npy']
random_label_names = ['Depth-Normalized Neural Net with KNN',\
					  'Chlorophyll-a',\
					  'nFLH',\
					  'Soto et al.',\
					  'Amin et al.']
date_files = ['roc_curve_info/'+configfilename+'.npy',\
			  'roc_curve_info/'+configfilename+'_Stumpf.npy',\
			  'roc_curve_info/'+configfilename+'_Shehhi.npy',\
			  'roc_curve_info/'+configfilename+'_Tomlinson.npy',\
			  'roc_curve_info/'+configfilename+'_Soto.npy',\
			  'roc_curve_info/'+configfilename+'_AminRBD.npy',\
			  'roc_curve_info/'+configfilename+'_AminRBDKBBI.npy',\
			  'roc_curve_info/'+configfilename+'_Cannizzaro2008.npy',\
			  'roc_curve_info/'+configfilename+'_Cannizzaro2009.npy'
			  ]
date_label_names = ['Neural Net with Remote Sensing and KNN Features',\
					'Stumpf et al. (2003)',\
					'nFLH',\
					'Tomlinson et al.',\
					'Soto et al.',\
					'Amin et al. (RBD)',\
					'Amin et al. (RBD+KBBI)',\
					'Cannizzaro et al. (2008)',\
					'Cannizzaro et al. (2009)'
					]

plotColors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
#plotNumber = 0

#plt.figure(dpi=500)
#plt.plot([0, 1], [0, 1], 'k--')
#plt.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], 'k')

#for file in random_files:
#	fpr_and_tprs = np.load(file)
#	if(len(fpr_and_tprs)>2):
#		fpr = fpr_and_tprs[:, 0]

#		tpr_means = np.zeros(fpr_and_tprs.shape[0])
#		tpr_stds = np.zeros(fpr_and_tprs.shape[0])
#		for i in range(fpr_and_tprs.shape[0]):
#			tpr_means[i] = np.mean(fpr_and_tprs[i, 1:])
#			tpr_stds[i] = np.std(fpr_and_tprs[i, 1:])
#		# Insert values to make sure plots start at (0, 0)
#		fpr = np.insert(fpr, 0, 0)
#		tpr_means = np.insert(tpr_means, 0, 0)
#		tpr_stds = np.insert(tpr_stds, 0, 0)
#		# Insert values to make sure plots end at (1, 1)
#		fpr = np.append(fpr, 1)
#		tpr_means = np.append(tpr_means, 1)
#		tpr_stds = np.append(tpr_stds, 1)
#		# margin of error for 95% confidence interval
#		# margin of error = z*(population standard deviation/sqrt(n))
#		# for 95% CI, z=1.96
#		tpr_moes = 1.96*(tpr_stds/(math.sqrt(21)))
#		plt.plot(fpr, tpr_means, label=random_label_names[plotNumber], color=plotColors[plotNumber], zorder=0)
#		x_values = np.concatenate((fpr, np.flip(fpr)))
#		y_values = np.concatenate((tpr_means+tpr_moes, np.flip(tpr_means-tpr_moes)))
#		plt.fill(x_values, y_values, alpha=0.3, color=plotColors[plotNumber], zorder=0)
#		plotNumber += 1
#	else:
#		fpr = fpr_and_tprs[0]
#		tpr = fpr_and_tprs[1]
#		plt.scatter(fpr, tpr, label=random_label_names[plotNumber], color=plotColors[plotNumber], zorder=10)
#		plotNumber += 1

#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.xlim(-0.05, 1.05)
#plt.ylim(-0.05, 1.05)
#plt.title('Random Split of Train/Test')
#plt.legend(loc='lower right')
#plt.savefig('roc_curve_plots/lit_compare_random.png', bbox_inches='tight')

plotNumber = 0

plt.figure(dpi=500)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], 'k')

for file in date_files:
	fpr_and_tprs = np.load(file)
	if(len(fpr_and_tprs)>2):
		fpr = fpr_and_tprs[:, 0]

		tpr_means = np.zeros(fpr_and_tprs.shape[0])
		tpr_stds = np.zeros(fpr_and_tprs.shape[0])
		for i in range(fpr_and_tprs.shape[0]):
			tpr_means[i] = np.mean(fpr_and_tprs[i, 1:])
			tpr_stds[i] = np.std(fpr_and_tprs[i, 1:])
		# Insert values to make sure plots start at (0, 0)
		fpr = np.insert(fpr, 0, 0)
		tpr_means = np.insert(tpr_means, 0, 0)
		tpr_stds = np.insert(tpr_stds, 0, 0)
		# Insert values to make sure plots end at (1, 1)
		fpr = np.append(fpr, 1)
		tpr_means = np.append(tpr_means, 1)
		tpr_stds = np.append(tpr_stds, 1)
		# margin of error for 95% confidence interval
		# margin of error = z*(population standard deviation/sqrt(n))
		# for 95% CI, z=1.96
		tpr_moes = (1.96*(tpr_stds/(math.sqrt(21))))/2
		plt.plot(fpr, tpr_means, label=date_label_names[plotNumber], color=plotColors[plotNumber], zorder=0)
		x_values = np.concatenate((fpr, np.flip(fpr)))
		y_values = np.concatenate((tpr_means+tpr_moes, np.flip(tpr_means-tpr_moes)))
		plt.fill(x_values, y_values, alpha=0.3, color=plotColors[plotNumber], zorder=0)
		plotNumber += 1
	else:
		fpr = fpr_and_tprs[0]
		tpr = fpr_and_tprs[1]
		plt.scatter(fpr, tpr, label=date_label_names[plotNumber], color=plotColors[plotNumber], zorder=10)
		plotNumber += 1

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.title('Comparison with Literature Methods at 50k cell/L')
plt.legend(loc='lower right')
plt.savefig('roc_curve_plots/lit_compare_date_50k_hill.png', bbox_inches='tight')