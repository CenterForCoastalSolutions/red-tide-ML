import numpy as np
import pickle
import math
import netCDF4
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
import skgstat as skg
import pandas as pd

def unique(list1):
	list_set = set(list1)
	unique_list = (list(list_set))
	return unique_list

def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx

florida_data = np.load('florida_data.npy')
florida_data_dates = np.load('florida_data_dates.npy')
florida_lats = np.load('florida_data_lats.npy')
florida_lons = np.load('florida_data_lons.npy')

summary_stats = {}

recompute_average_imagery = False

if(recompute_average_imagery == True):

	florida_data_dates_no_time = []
	for i in range(len(florida_data_dates)):
		date_string = np.datetime_as_string(florida_data_dates[i], unit='D')
		florida_data_dates_no_time.append(date_string)
	florida_data_dates_no_time_unique = unique(florida_data_dates_no_time)
	florida_data_dates_no_time_unique.sort()

	#Average images from the same date
	florida_data_per_date = np.zeros((len(florida_data_dates_no_time_unique), florida_data.shape[1], florida_data.shape[2]))
	for i in range(len(florida_data_dates_no_time_unique)):
		date_inds = [j for j in range(len(florida_data_dates_no_time)) if florida_data_dates_no_time[j] == florida_data_dates_no_time_unique[i]]
		
		for j in range(florida_data.shape[1]):
			for k in range(florida_data.shape[2]):
				valid_count = 0
				for ind in date_inds:
					if(florida_data[ind, j, k] > -0.5):
						florida_data_per_date[i, j, k] += florida_data[ind, j, k]
						valid_count += 1
				if(valid_count>0):
					florida_data_per_date[i, j, k] /= valid_count
				else:
					florida_data_per_date[i, j, k] = -1

	florida_data = florida_data_per_date
	florida_data_dates = florida_data_dates_no_time_unique

	florida_data_sum = np.zeros_like(florida_data)
	florida_data_count = np.zeros_like(florida_data)
	florida_data_average = np.zeros_like(florida_data)

	#Average images from 14 days on each side
	for i in range(len(florida_data_dates)):
		for j in range(len(florida_data_dates)):
			days_apart = abs(np.datetime64(florida_data_dates[i])-np.datetime64(florida_data_dates[j])).astype(int)

			if(days_apart < 14):
				for k in range(florida_data.shape[1]):
					for l in range(florida_data.shape[2]):
						if(florida_data[j, k, l] > -0.5):
							florida_data_sum[i, k, l] += florida_data[j, k, l]
							florida_data_count[i, k, l] += 1

	for i in range(len(florida_data_dates)):
		for k in range(florida_data.shape[1]):
			for l in range(florida_data.shape[2]):
				if(florida_data_count[i, k, l] > 0):
					florida_data_average[i, k, l] = florida_data_sum[i, k, l]/florida_data_count[i, k, l]
				else:
					florida_data_average[i, k, l] = -1

	for i in range(len(florida_data_dates)):
		plt.figure(dpi=500)
		plt.imshow(np.squeeze(florida_data_average[i, :, :]).T)
		plt.gca().invert_yaxis()
		plt.title(florida_data_dates[i])
		plt.clim(0, 5)
		plt.colorbar()
		plt.savefig('florida_data_plots_average/extracted'+str(i).zfill(5)+'.png', bbox_inches='tight')
		plt.close()

	np.save('florida_data_average.npy', florida_data_average)
	np.save('florida_data_average_dates.npy', florida_data_dates)
else:
	florida_data_dates = np.load('florida_data_average_dates.npy')
	florida_data_average = np.load('florida_data_average.npy')

# Size (days, lon, lat)
florida_data = florida_data_average

date_start = 730
date_end = 760

data_netcdf4 = netCDF4.Dataset('/run/media/rfick/UF10/HYCOM/hycom_correlations.nc', mode='r')

data_xarray = xr.open_dataset(xr.backends.NetCDF4DataStore(data_netcdf4))

data_xarray_chlor_a = np.zeros(data_xarray['time'].shape)

florida_data_dates_datetime64 = np.empty(len(florida_data_dates), dtype='datetime64[s]')

for i in range(len(florida_data_dates_datetime64)):
	florida_data_dates_datetime64[i] = np.datetime64(florida_data_dates[i])


track_starts_lats = data_xarray['lat'].values[:, 0]
track_starts_lons = data_xarray['lon'].values[:, 0]

# Find chlor value at each day at the start of each track for variogram
track_starts_per_day = np.zeros((data_xarray['time'].shape[0], date_end-date_start))
for i in range(track_starts_per_day.shape[0]):
	print('Processing track {}/{}'.format(i, data_xarray_chlor_a.shape[0]))
	sampleLat = data_xarray['lat'].values[i, 0]
	sampleLon = data_xarray['lon'].values[i, 0]

	closestLonInd = find_nearest(florida_lons, sampleLon)
	closestLatInd = find_nearest(florida_lats, sampleLat)
	for j in range(track_starts_per_day.shape[1]):
		track_starts_per_day[i, j] = florida_data[date_start + j, closestLonInd, closestLatInd]
np.save('track_starts_per_day.npy', track_starts_per_day)
np.save('track_starts_lats.npy', track_starts_lats)
np.save('track_starts_lons.npy', track_starts_lons)
asdf



for i in range(data_xarray_chlor_a.shape[0]):
	print('Processing track {}/{}'.format(i, data_xarray_chlor_a.shape[0]))
	for j in range(data_xarray_chlor_a.shape[1]):
		sampleTime = data_xarray['time'].values[i, j]
		sampleLat = data_xarray['lat'].values[i, j]
		sampleLon = data_xarray['lon'].values[i, j]
		timedifference = (np.datetime64(sampleTime) - florida_data_dates_datetime64[date_start:date_end])/np.timedelta64(1, 's')
		closestTimeInd = np.argmin(np.abs(timedifference))

		closestLonInd = find_nearest(florida_lons, sampleLon)
		closestLatInd = find_nearest(florida_lats, sampleLat)

		#find correct physical inds by looking at data_xarray['lat'] and data_xarray['lon']
		#save correct chlor pixel
		data_xarray_chlor_a[i, j] = florida_data[date_start + closestTimeInd, closestLonInd, closestLatInd]

np.save('data_xarray_chlor_a.npy', data_xarray_chlor_a)
alsdkjf

outputdt = dt.timedelta(hours=1)
timerange = np.arange(np.nanmin(data_xarray['time'].values),
                      np.nanmax(data_xarray['time'].values)+np.timedelta64(outputdt), 
                      outputdt)  # timerange in nanoseconds

chlor_diffs = []

for i in range(date_start, date_end):
	print('Processing day {}/{}'.format(i-date_start, date_end-date_start))

	timedifference = (timerange - florida_data_dates[i])/np.timedelta64(1, 's')
	closestTime = np.argmin(np.abs(timedifference))
	time_id = np.where(data_xarray['time'] == timerange[closestTime]) # Indices of the data where time = 0

	lon_values = data_xarray['lon'].values[time_id[0], :]
	lat_values = data_xarray['lat'].values[time_id[0], :]
	time_values = data_xarray['time'].values[time_id[0], :]
	traj_end_inds = (~np.isnan(lon_values)).cumsum(1).argmax(1)

	for j in range(lon_values.shape[0]):
		start_time = time_values[time_id[0][j], time_id[1][j]]
		end_time = time_values[time_id[0][j], traj_end_inds[j]]
		if(np.isnan(start_time)==False and np.isnan(end_time)==False):
			days_apart = int((end_time - start_time)/np.timedelta64(1, 'D'))

			lon_start_ind = find_nearest(florida_lons, lon_values[time_id[0][j], time_id[1][j]])
			lat_start_ind = find_nearest(florida_lats, lat_values[time_id[0][j], time_id[1][j]])
			lon_end_ind = find_nearest(florida_lons, lon_values[time_id[0][j], traj_end_inds[j]])
			lat_end_ind = find_nearest(florida_lats, lat_values[time_id[0][j], traj_end_inds[j]])
			chlor_start = florida_data[i, lon_start_ind, lat_start_ind]
			chlor_end = florida_data[i+days_apart, lon_end_ind, lat_end_ind]
			lon_start = lon_values[time_id[0][j], time_id[1][j]]
			lon_end = lon_values[time_id[0][j], traj_end_inds[j]]
			lat_start = lat_values[time_id[0][j], time_id[1][j]]
			lat_end = lat_values[time_id[0][j], traj_end_inds[j]]


			chlor_diffs.append([abs(chlor_end-chlor_start), math.sqrt((lon_end - lon_start)**2 + (lat_end - lat_start)**2), days_apart])

chlor_diffs = np.asarray(chlor_diffs)

np.save('chlor_diffs.npy', chlor_diffs)

fig = plt.figure(dpi=500)
ax = fig.add_subplot(projection='3d')
ax.scatter(chlor_diffs[:, 1], chlor_diffs[:, 2], chlor_diffs[:, 0])
ax.set_xlabel('Space')
ax.set_ylabel('Time')
ax.set_zlabel('Chlorophyll-a Difference')
plt.savefig('variogram_scatter.png', bbox_inches='tight')
plt.close()

coordinates = []
values = []

for i in range(florida_data.shape[1]):
	for j in range(florida_data.shape[2]):
		if(florida_data[0, i, j] > -0.5):
			coordinates.append([florida_lons[i], florida_lats[i]])
			values.append(np.squeeze(florida_data[15:45, i, j]))

coordinates = np.asarray(coordinates)
values = np.asarray(values)

#Only use a subset of points to limit memory usage
value_perm = np.random.permutation(values.shape[0])

# Coordinates is (m, n) - m locations with n dimensions
# Values is (m, t) - m locations with t time steps
V = skg.SpaceTimeVariogram(coordinates=coordinates[value_perm[:1000], :], values=values[value_perm[:1000], :])

V.surface()
plt.savefig('variogram.png', bbox_inches='tight')
plt.close()

#for i in range(florida_data.shape[0]):
#	for j in range(i+1, florida_data.shape[0]):
#		print('Processing image pair {} {}'.format(i, j))

#		#days_apart = int(abs((florida_data_dates[i]-florida_data_dates[j])/np.timedelta64(1, 'D')))
#		days_apart = abs(np.datetime64(florida_data_dates[i])-np.datetime64(florida_data_dates[j])).astype(int)

#		for k in range(florida_data.shape[1]):
#			for l in range(florida_data.shape[2]):
#				for m in range(florida_data.shape[1]):
#					for n in range(florida_data.shape[2]):
#						# Both images have valid pixel at the same location
#						if(florida_data[i, k, l] > -0.5 and florida_data[j, m, n] > -0.5):
#							pixel_dist = int(np.sqrt((k-m)**2 + (l-n)**2))

#							if(str(pixel_dist)+'_'+str(days_apart)+'_x' in summary_stats.keys()):
#								summary_stats[str(pixel_dist)+'_'+str(days_apart)+'_x'] = summary_stats[str(pixel_dist)+'_'+str(days_apart)+'_x'] + abs(florida_data[i, k, l]-florida_data[j, m, n])
#								summary_stats[str(pixel_dist)+'_'+str(days_apart)+'_n'] = summary_stats[str(pixel_dist)+'_'+str(days_apart)+'_n'] + 1
#							else:
#								summary_stats[str(pixel_dist)+'_'+str(days_apart)+'_x'] = abs(florida_data[i, k, l]-florida_data[j, k, l])
#								summary_stats[str(pixel_dist)+'_'+str(days_apart)+'_n'] = 1

#with open('florida_data_summary_stats.pkl', 'wb') as f:
#	pickle.dump(summary_stats, f)

#meanValue = np.zeros((int(len(summary_stats.keys())/2)))
#count = np.zeros((int(len(summary_stats.keys())/2)))
#daysAhead = np.zeros((int(len(summary_stats.keys())/2)))

#for i in range(1, meanValue.shape[0]+1):
#	meanValue[i-1] = summary_stats[str(i)+'_x']/summary_stats[str(i)+'_n']
#	count[i-1] = summary_stats[str(i)+'_n']
#	daysAhead[i-1] = i

#plt.figure(dpi=500)
#plt.plot(daysAhead, meanValue)
#plt.xlabel('Days Ahead')
#plt.ylabel('Average Chlor_a Difference')
#plt.title('Chlor_a vs Time')
#plt.savefig('chlor_a_vs_time_mean.png')

#print(count)

#plt.figure(dpi=500)
#plt.plot(daysAhead, count)
#plt.xlabel('Days Ahead')
#plt.ylabel('Pixel Count')
#plt.title('Chlor_a vs Time')
#plt.savefig('chlor_a_vs_time_count.png')