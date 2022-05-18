import numpy as np

# Implements red tide detection method from Soto et al
# https://ioccg.org/wp-content/uploads/2021/05/ioccg_report_20-habs-2021-web.pdf (Chapter 6)
# features are in order ['chl_ocx', 'nflh', 'Rrs_443', 'Rrs_555']
def SotoEtAlDetector(features):
	#bbp from Morel 1988
	eps = 0.000000001
	# add eps to chlorophyll values to avoid divide by zero issues with log(chl)
	chl = features[:, 0] + eps
	bbp_morel = 0.3 * np.power(chl, 0.62) * (0.002 + 0.02*(0.5 - 0.25*np.log10(chl)))

	#bbp_555 from Carder et al. 1999
	bbp_555 = 2.058*features[:, 2] - 0.00182

	bbp_555_ratio = bbp_555/bbp_morel

	red_tide = np.zeros_like(bbp_555)
	red_tide_inds = np.where((features[:, 0]>1.5) & (features[:, 1]>0.1) & (bbp_555_ratio<1))
	red_tide[red_tide_inds[0]] = 1

	return red_tide