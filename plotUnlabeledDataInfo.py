import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

features = ['par', 'Kd_490', 'chlor_a', 'Rrs_443', 'Rrs_469', 'Rrs_488', 'nflh', 'KNN Estimate']

nonValidationDataWUnlabeled = np.load('nonValidationDataWUnlabeled.npy')
nonValidationTargetsWUnlabeled = np.load('nonValidationTargetsWUnlabeled.npy')

predPositives = nonValidationDataWUnlabeled[-10:,:]
predNegatives = nonValidationDataWUnlabeled[-100:-10,:]
trueNegativeInds = np.where(nonValidationTargetsWUnlabeled[:-100,0]==1)[0]
truePositiveInds = np.where(nonValidationTargetsWUnlabeled[:-100,1]==1)[0]
trueNegatives = nonValidationDataWUnlabeled[trueNegativeInds,:]
truePositives = nonValidationDataWUnlabeled[truePositiveInds,:]

for i in range(len(features)):
	plt.figure(dpi=500)
	plt.hist(trueNegatives[:, i], bins=50, alpha=0.5, density=1, color='b', label='True Negatives')
	plt.hist(truePositives[:, i], bins=50, alpha=0.5, density=1, color='r', label='True Positives')
	plt.hist(predNegatives[:, i], bins=50, alpha=0.5, density=1, color='c', label='Predicted Negatives')
	plt.hist(predPositives[:, i], bins=50, alpha=0.5, density=1, color='m', label='Predicted Positives')
	plt.legend()
	plt.title(features[i])
	plt.savefig('unlabeledFeatureInfo/{}.png'.format(features[i]))

pca = PCA(n_components=3)
nonValidationDataWUnlabeledPCA = pca.fit_transform(nonValidationDataWUnlabeled)

predPositivesPCA = nonValidationDataWUnlabeledPCA[-1000:,:]
predNegativesPCA = nonValidationDataWUnlabeledPCA[-10000:-1000,:]
trueNegativesPCA = nonValidationDataWUnlabeledPCA[trueNegativeInds,:]
truePositivesPCA = nonValidationDataWUnlabeledPCA[truePositiveInds,:]

plt.figure(dpi=500)
plt.scatter(trueNegativesPCA[:, 0], trueNegativesPCA[:, 1], color='b', label='True Negatives')
plt.scatter(truePositivesPCA[:, 0], truePositivesPCA[:, 1], color='r', label='True Positives')
plt.legend()
plt.savefig('unlabeledFeatureInfo/PCAplotTrue.png')

plt.figure(dpi=500)
plt.scatter(trueNegativesPCA[:, 0], trueNegativesPCA[:, 1], color='b', label='True Negatives')
plt.scatter(truePositivesPCA[:, 0], truePositivesPCA[:, 1], color='r', label='True Positives')
plt.scatter(predNegativesPCA[:, 0], predNegativesPCA[:, 1], color='c', label='Predicted Negatives')
plt.scatter(predPositivesPCA[:, 0], predPositivesPCA[:, 1], color='m', label='Predicted Positives')
plt.legend()
plt.savefig('unlabeledFeatureInfo/PCAplot.png')