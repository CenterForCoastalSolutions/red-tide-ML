import numpy as np
from detectors.loadf0Data import *
from utils import *

# Amin, R., et al. "MODIS and MERIS detection of dinoflagellates blooms using the RBD technique."
# Remote Sensing of the Ocean, Sea Ice, and Large Water Regions 2009. Vol. 7473. SPIE, 2009.
# This method detects red tide by the following thresholds:
# RBD > 0.15 W/(m^2)/microm/sr
# features in order are ['Rrs_667', 'Rrs_678']
def AminEtAlRBDDetector(features):
	wv, solarIrradiance = loadf0Data()

	idx667 = find_nearest(wv, 667)
	idx678 = find_nearest(wv, 678)

	# Convert remote sensing reflectance values to water leaving radiance
	features[:, 0] = solarIrradiance[idx667]*features[:, 0]
	features[:, 1] = solarIrradiance[idx678]*features[:, 1]

	RBD = features[:, 1] - features[:, 0]

	return RBD