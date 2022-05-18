import numpy as np
from detectors.loadf0Data import *
from utils import *

# Amin, Ruhul, et al. "Noval optical techniques for detecting and classifying toxic dinoflagellate Karenia brevis blooms using satellite imagery."
# Optics express 17.11 (2009): 9126-9144.
# This method detects red tide by the following thresholds:
# RBD > 0.15 W/(m^2)/microm/sr
# KBBI > 0.3*RBD
# features in order are ['Rrs_667', 'Rrs_678']
def AminEtAlRBDKBBIDetector(features):
	wv, solarIrradiance = loadf0Data()

	idx667 = find_nearest(wv, 667)
	idx678 = find_nearest(wv, 678)

	# Convert remote sensing reflectance values to water leaving radiance
	features[:, 0] = solarIrradiance[idx667]*features[:, 0]
	features[:, 1] = solarIrradiance[idx678]*features[:, 1]

	RBD = features[:, 1] - features[:, 0]
	KBBI = (features[:, 1] - features[:, 0])/(features[:, 1] + features[:, 0])

	red_tide = np.zeros_like(RBD)

	for i in range(len(red_tide)):
		if(RBD[i]>0.015 and KBBI[i]>0.3*RBD[i]):
			red_tide[i] = 1

	return red_tide