import numpy as np

def find_day_ind(array, value):
	idx = 0
	for i in range(len(array)):
		if(array[i].year == value.year and array[i].month == value.month and array[i].day == value.day):
			idx = i
	return idx

# Stumpf, R. P., Culver, M. E., Tester, P. A., Tomlinson, M., Kirkpatrick, G. J., Pederson, B. A., ... & Soracco, M. (2003).
# Monitoring Karenia brevis blooms in the Gulf of Mexico using satellite ocean color imagery and other data. Harmful Algae, 2(2), 147-160.
# This method detects red tide by the following thresholds:
# chlorophyll-a > 1 ug/L above the mean of chlorophyll imagery from 2 months to 2 weeks ago
# features in order are ['Sample Date', 'chlor_a']
# MODIS chlorophyll-a is provided in units of mg/(m^3) which is equivalent to ug/L
def StumpfEtAlDetector(features):

	computedDates = np.load('/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/Stumpf_Chlor_Anomaly_Info/computedDates.npy', allow_pickle=True)
	computedChlorAvg = np.load('/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/Stumpf_Chlor_Anomaly_Info/computedChlorAvg.npy')
	
	chlorMinusAvg = np.zeros((features.shape[0], 1))

	for i in range(features.shape[0]):
		computedInd = find_day_ind(computedDates, features[i, 0])

		chlorMinusAvg[i] = features[i, 1] - computedChlorAvg[computedInd]

	return chlorMinusAvg