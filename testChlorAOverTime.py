import numpy as np
import os
import netCDF4
import pandas as pd
import datetime as dt
import xarray as xr
import matplotlib.pyplot as plt

# a = line point segment 1, b = line point segment 2, c = point to check
def isLeft(a_x, a_y, b_x, b_y, c_x, c_y):
	return ((b_x - a_x)*(c_y - a_y) - (b_y - a_y)*(c_x - a_x)) > 0

def find_nearest(lat_array, lat_value, lon_array, lon_value):
	lat_array = np.asarray(lat_array)
	lon_array = np.asarray(lon_array)
	idx = (np.abs(lon_array - lon_value) + np.abs(lat_array - lat_value)).argmin()
	return idx, np.min((np.abs(lon_array - lon_value) + np.abs(lat_array - lat_value)))

startDate = pd.Timestamp(year=2016, month=6, day=1, hour=0)
endDate = pd.Timestamp(year=2019, month=6, day=1, hour=0)

data_folder = '/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/requested_files'
data_list = os.listdir(data_folder)

reduced_file_list = []
for i in range(len(data_list)):
	year = int(data_list[i][1:5])
	if(abs(startDate.year - year) < 1.5 or abs(endDate.year - year) < 1.5):
		reduced_file_list.append(data_list[i])

file_path_list = []
date_list = []

for i in range(len(reduced_file_list)):
	print('Processing files: {}/{}'.format(i, len(reduced_file_list)))

	file_id = reduced_file_list[i]
	file_path = data_folder + '/' + file_id

	fh = netCDF4.Dataset(file_path, mode='r')
	collectionDateTime = fh.time_coverage_start
	year = int(collectionDateTime[0:4])
	month = int(collectionDateTime[5:7])
	day = int(collectionDateTime[8:10])
	hour = int(collectionDateTime[11:13])
	minute = int(collectionDateTime[14:16])
	second = int(collectionDateTime[17:19])
	collectionTimeStamp = dt.datetime(year, month, day, hour, minute, second)

	if(collectionTimeStamp >= startDate and collectionTimeStamp < endDate):
		file_path_list.append(file_path)
		date_list.append(collectionTimeStamp)

li=[]
for i in range(len(date_list)):
	li.append([date_list[i],i])
li.sort()
sort_index = []
for x in li:
	sort_index.append(x[1])

date_list_sorted = [date_list[i] for i in sort_index]
file_path_list_sorted = [file_path_list[i] for i in sort_index]

step_size = 0.05
florida_x = np.arange(-92, -75, step_size)
florida_y = np.arange(20, 35, step_size)
florida_lats = np.tile(florida_y, len(florida_x))
florida_lons = np.repeat(florida_x, len(florida_y))
florida_lats = np.reshape(florida_lats, (florida_x.shape[0], florida_y.shape[0]), order='C')
florida_lons = np.reshape(florida_lons, (florida_x.shape[0], florida_y.shape[0]), order='C')

min_lat_idx = np.argmin(np.abs(florida_lats[0, :] - 24.5))
max_lat_idx = np.argmin(np.abs(florida_lats[0, :] - 28.7))
min_lon_idx = np.argmin(np.abs(florida_lons[:, 0] - -84.2))
max_lon_idx = np.argmin(np.abs(florida_lons[:, 0] - -80.3))

florida_lats = florida_lats[min_lon_idx:max_lon_idx, min_lat_idx:max_lat_idx]
florida_lons = florida_lons[min_lon_idx:max_lon_idx, min_lat_idx:max_lat_idx]

florida_lats = florida_lats[0, :]
florida_lons = florida_lons[:, 0]

np.save('florida_data_lats.npy', florida_lats)
np.save('florida_data_lons.npy', florida_lons)

florida_data = -1*np.ones((len(file_path_list_sorted), florida_lons.shape[0], florida_lats.shape[0]))
florida_data_dates = []

# Line points of bounds to check
point1_x = 50
point1_y = 0
point2_x = 10
point2_y = 83

point3_x = 70
point3_y = 0
point4_x = 42
point4_y = 83

for i in range(len(file_path_list_sorted)):
	print('Processing files: {}/{}'.format(i, len(file_path_list_sorted)))

	fh = netCDF4.Dataset(file_path_list_sorted[i], mode='r')
	collectionDateTime = fh.time_coverage_start
	year = int(collectionDateTime[0:4])
	month = int(collectionDateTime[5:7])
	day = int(collectionDateTime[8:10])
	hour = int(collectionDateTime[11:13])
	minute = int(collectionDateTime[14:16])
	second = int(collectionDateTime[17:19])
	collectionTimeStamp = dt.datetime(year, month, day, hour, minute, second)

	collectionTimeStamp = np.datetime64(collectionTimeStamp)
	florida_data_dates.append(collectionTimeStamp)

	nav_dataset = xr.open_dataset(file_path_list_sorted[i], 'navigation_data')

	latitude = nav_dataset['latitude']
	longitude = nav_dataset['longitude']
	latarr = np.array(latitude).flatten()
	longarr = np.array(longitude).flatten()

	dataset = xr.open_dataset(file_path_list_sorted[i], 'geophysical_data')
	chlor_a = np.array(dataset['chlor_a']).flatten()

	for j in range(len(longarr)):
		chlor_a_new = chlor_a[j]

		if(np.isnan(chlor_a_new) == False):
			lat_new = latarr[j]
			lon_new = longarr[j]
			lat_new_idx = np.argmin(np.abs(florida_lats-lat_new))
			lon_new_idx = np.argmin(np.abs(florida_lons-lon_new))
			if(np.sqrt(np.abs(lon_new - florida_lons[lon_new_idx]) + np.abs(lat_new - florida_lats[lat_new_idx])) < 0.15):
				#if(isLeft(point1_x, point1_y, point2_x, point2_y, lon_new_idx, lat_new_idx) == False and isLeft(point3_x, point3_y, point4_x, point4_y, lon_new_idx, lat_new_idx) == True):
				if(isLeft(point3_x, point3_y, point4_x, point4_y, lon_new_idx, lat_new_idx) == True):
					florida_data[i, lon_new_idx, lat_new_idx] = chlor_a_new

	plt.figure(dpi=500)
	plt.imshow(np.squeeze(florida_data[i, :, :]))
	plt.clim(0, 5)
	plt.colorbar()
	plt.savefig('florida_data_plots/extracted{}.png'.format(i))

np.save('florida_data.npy', florida_data)
np.save('florida_data_dates.npy', florida_data_dates)