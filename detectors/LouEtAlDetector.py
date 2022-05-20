import numpy as np

# Implements red tide detection method from
# Lou, Xiulin, and Chuanmin Hu. "Diurnal changes of a harmful algal bloom
# in the East China Sea: Observations from GOCI." Remote Sensing of Environment 140 (2014): 562-572.
# features are in order ['Rrs_443', 'Rrs_488', 'Rrs_555']
def LouEtAlDetector(features):
	# RI = (Rrs_555 - Rrs_443)/(Rrs_488 - Rrs_443)
	RI = np.zeros(features.shape[0])
	for i in range(features.shape[0]):
		if((features[i, 1] - features[i, 0]) == 0):
			RI[i] = 0
		else:
			RI[i] = (features[i, 2] - features[i, 0])/(features[i, 1] - features[i, 0])

	return RI