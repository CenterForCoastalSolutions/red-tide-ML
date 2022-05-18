import numpy as np

# Thuillier, G., Herse, M., Labs, D., et al. (2003). The solar spectral irradiance from 200 to 2400 nm as measured by the
# SOLSPEC spectrometer from the Atlas and Eureca missions. Solar Physics 214: 1. doi:10.1023/A:1024048429145
# From: https://oceancolor.gsfc.nasa.gov/docs/rsr/f0.txt
def loadf0Data():

    lines = []
    with open('detectors/f0.txt') as f:
        lines = f.readlines()

    wv = []
    solarIrradiance = []

    for line_number in range(15, len(lines)):
        x = lines[line_number].split(" ")
        wv.append(int(x[0]))
        solarIrradiance.append(float(x[1]))

    wv = np.array(wv)
    solarIrradiance = np.array(solarIrradiance)

    return wv, solarIrradiance