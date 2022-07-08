import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

startDate = pd.Timestamp(year=2018, month=7, day=5, hour=0)
endDate = pd.Timestamp(year=2018, month=10, day=31, hour=0)

dates = []

testDate = startDate
while(testDate <= endDate):
	dates.append(testDate)
	testDate = testDate + pd.Timedelta(days=1)

day_counter = 0
for day in dates:
	chlor_a_map = np.load('chlor_maps/chlor_image{}.npy'.format(day_counter))

	plt.figure(dpi=500)
	plt.imshow(chlor_a_map.T)
	plt.clim(-1, 10)
	plt.colorbar()
	plt.title('Chlor-a Concentration {}/{}/{}'.format(day.month, day.day, day.year))
	plt.gca().invert_yaxis()
	plt.savefig('chlor_maps/chlor_image{}.png'.format(str(day_counter).zfill(5)), bbox_inches='tight')

	day_counter += 1