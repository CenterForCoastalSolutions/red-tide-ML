import numpy as np
from detectors.loadf0Data import *
from utils import *

# Implements red tide detection method from Tomlinson et al:
# Tomlinson, M. C., T. T. Wynne, and R. P. Stumpf. "An evaluation of remote
# sensing techniques for enhanced detection of the toxic dinoflagellate,
# karenia brevis." Remote Sensing of Environment 113.3 (2009): 598-609
# features in order are ['Rrs_443', 'Rrs_488', 'Rrs_531']
def Tomlinson2009EtAlDetector(features):
	wv, solarIrradiance = loadf0Data()

	idx443 = find_nearest(wv, 443)
	idx488 = find_nearest(wv, 488)
	idx531 = find_nearest(wv, 531)

	# Convert remote sensing reflectance values to water leaving radiance
	features[:, 0] = solarIrradiance[idx443]*features[:, 0]
	features[:, 1] = solarIrradiance[idx488]*features[:, 1]
	features[:, 2] = solarIrradiance[idx531]*features[:, 2]

	SS_490 = features[:, 1] - features[:, 0] - (features[:, 2] - features[:, 0])*((488 - 443)/(531 - 443))

	# Multiply by -1 because negative values are indicative of blooms
	SS_490 = -1*SS_490

	return SS_490