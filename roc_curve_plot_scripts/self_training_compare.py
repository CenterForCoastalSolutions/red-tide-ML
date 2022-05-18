import numpy as np
import matplotlib.pyplot as plt
import math

meanAUCs = []

for i in range(101):
	file = 'roc_curve_info/date_train_test_depth_norm_w_knn_SelfTraining{}Iter.npy'.format(i)
	fpr_and_tprs = np.load(file)
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

	# Calculate mean AUC
	meanAUC = 0
	for i in range(len(tpr_means)-1):
		meanAUC = meanAUC + ((fpr[i+1]-fpr[i])*(tpr_means[i])) + ((fpr[i+1]-fpr[i])*((tpr_means[i+1]-tpr_means[i])/2))

	meanAUCs.append(meanAUC)

plt.figure(dpi=500)
plt.plot(range(101), meanAUCs)
plt.xlabel('Number of Self-Training Iterations')
plt.ylabel('Mean ROC AUC')
plt.title('Model Performance over Self-Training Iterations')
plt.savefig('roc_curve_plots/self_training_compare.png', bbox_inches='tight')