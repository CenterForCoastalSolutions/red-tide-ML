import numpy as np
import pandas as pd
import math
import os
import datetime as dt
import netCDF4
import xarray as xr
import matplotlib.pyplot as plt
from findMatrixCoordsBedrock import *

def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx

paired_df = pd.read_pickle('paired_dataset.pkl')

findValidMask = True

if(findValidMask):
	# Sample lat/lon so we only take unlabeled pixels within a certain distance of labeled pixels
	paired_longitude = paired_df['Longitude'].to_numpy().copy()
	paired_latitude = paired_df['Latitude'].to_numpy().copy()

	step_size = 0.05
	florida_x = np.arange(-92, -75, step_size)
	florida_y = np.arange(20, 35, step_size)
	florida_lats = np.tile(florida_y, len(florida_x))
	florida_lons = np.repeat(florida_x, len(florida_y))
	florida_lats = np.reshape(florida_lats, (florida_x.shape[0], florida_y.shape[0]), order='C')
	florida_lons = np.reshape(florida_lons, (florida_x.shape[0], florida_y.shape[0]), order='C')

	florida_valid_mask = np.zeros((florida_lons.shape[0], florida_lats.shape[1]))

	# 0.1 in lat/lon units equates to roughly 10 km
	validDistance = 0.1

	#Mark valid pixels in mask that are close enough to each point
	for i in range(paired_longitude.shape[0]):
		if(i%100 == 0):
			print('Processing paired pixel {}/{}'.format(i, paired_longitude.shape[0]))

		point_lon = paired_longitude[i]
		point_lat = paired_latitude[i]

		for lon_ind in range(florida_valid_mask.shape[0]):
			for lat_ind in range(florida_valid_mask.shape[1]):
				pixel_lon = florida_lons[lon_ind, lat_ind]
				pixel_lat = florida_lats[lon_ind, lat_ind]

				pixelDist = math.sqrt((pixel_lon - point_lon)**2 + (pixel_lat - point_lat)**2)

				if(pixelDist <= validDistance):
					florida_valid_mask[lon_ind, lat_ind] = 1




	depth_lons = np.load('florida_x.npy')
	depth_lats = np.load('florida_y.npy')
	depth_florida_z = np.load('latlon_stats/angstrom_mean.npy')

	for lon_ind in range(florida_valid_mask.shape[0]):
		for lat_ind in range(florida_valid_mask.shape[1]):
			pixel_lon = florida_lons[lon_ind, lat_ind]
			pixel_lat = florida_lats[lon_ind, lat_ind]

			depth_lon_ind = find_nearest(depth_lons, pixel_lon)
			depth_lat_ind = find_nearest(depth_lats, pixel_lat)

			if(depth_florida_z[depth_lat_ind][depth_lon_ind] != -1):
				florida_valid_mask[lon_ind, lat_ind] = 0.5

	florida_valid_mask = florida_valid_mask[100:270, 60:200]

	plt.figure(dpi=500)
	plt.imshow(florida_valid_mask.T)
	plt.gca().invert_yaxis()
	plt.title('Unlabeled Pixel Mask')
	plt.savefig('unlabeled_pixels.png')

	asdfasd



	np.save('unlabeledPixels/florida_x.npy', florida_x)
	np.save('unlabeledPixels/florida_y.npy', florida_y)
	np.save('unlabeledPixels/florida_valid_mask.npy', florida_valid_mask)
else:
	florida_x = np.load('unlabeledPixels/florida_x.npy')
	florida_y = np.load('unlabeledPixels/florida_y.npy')
	florida_valid_mask = np.load('unlabeledPixels/florida_valid_mask.npy')

startDate = pd.Timestamp(year=2000, month=1, day=1, hour=0)
endDate = pd.Timestamp(year=2020, month=12, day=1, hour=0)

data_folder = '/run/media/rfick/UF10/MODIS-OC/MODIS-OC-data/requested_files'
data_list = os.listdir(data_folder)

reduced_file_list = []
for i in range(len(data_list)):
	year = int(data_list[i][1:5])
	if(abs(startDate.year - year) < 1.5 or abs(endDate.year - year) < 1.5 or (startDate.year < year and endDate.year > year)):
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

numPixels = 0

#Only take some data at random so the files aren't huge
perc_Data = 0.1

unlabeled_dataset = []

bedrock = netCDF4.Dataset('ETOPO1_Bed_g_gmt4.grd')
bedrock_x = bedrock['x'][:]
bedrock_y = bedrock['y'][:]
bedrock_z = bedrock['z'][:]

for i in range(len(file_path_list_sorted)):
	print('Processing files: {}/{}'.format(i, len(file_path_list_sorted)))

	fh = netCDF4.Dataset(file_path_list_sorted[i], mode='r')
	collectionDateTime = fh.time_coverage_start

	nav_dataset = xr.open_dataset(file_path_list_sorted[i], 'navigation_data')

	latitude = nav_dataset['latitude']
	longitude = nav_dataset['longitude']
	latarr = np.array(latitude).flatten()
	longarr = np.array(longitude).flatten()

	dataset = xr.open_dataset(file_path_list_sorted[i], 'geophysical_data')
	aot_869 = np.array(dataset['aot_869']).flatten()
	angstrom = np.array(dataset['angstrom']).flatten()
	Rrs_412 = np.array(dataset['Rrs_412']).flatten()
	Rrs_443 = np.array(dataset['Rrs_443']).flatten()
	Rrs_469 = np.array(dataset['Rrs_469']).flatten()
	Rrs_488 = np.array(dataset['Rrs_488']).flatten()
	Rrs_531 = np.array(dataset['Rrs_531']).flatten()
	Rrs_547 = np.array(dataset['Rrs_547']).flatten()
	Rrs_555 = np.array(dataset['Rrs_555']).flatten()
	Rrs_645 = np.array(dataset['Rrs_645']).flatten()
	Rrs_667 = np.array(dataset['Rrs_667']).flatten()
	Rrs_678 = np.array(dataset['Rrs_678']).flatten()
	chlor_a = np.array(dataset['chlor_a']).flatten()
	chl_ocx = np.array(dataset['chl_ocx']).flatten()
	Kd_490 = np.array(dataset['Kd_490']).flatten()
	pic = np.array(dataset['pic']).flatten()
	poc = np.array(dataset['poc']).flatten()
	ipar = np.array(dataset['ipar']).flatten()
	nflh = np.array(dataset['nflh']).flatten()
	par = np.array(dataset['par']).flatten()

	for j in range(len(longarr)):
		closestLon_ind = find_nearest(florida_x, longarr[j])
		closestLat_ind = find_nearest(florida_y, latarr[j])

		# If chlorophyll value is valid and pixel is close enough to 
		if(np.isnan(chlor_a[j]) == False and florida_valid_mask[closestLon_ind, closestLat_ind] == 1):
			if(np.random.rand() < perc_Data):
				bedrock_index0 = find_nearest(bedrock_y, latarr[j])
				bedrock_index1 = find_nearest(bedrock_x, longarr[j])

				sample_info = [collectionDateTime, latarr[j], longarr[j], aot_869[j], angstrom[j], Rrs_412[j],\
				Rrs_443[j], Rrs_469[j], Rrs_488[j], Rrs_531[j], Rrs_547[j], Rrs_555[j], Rrs_645[j], Rrs_667[j],\
				Rrs_678[j], chlor_a[j], chl_ocx[j], Kd_490[j], pic[j], poc[j], ipar[j], nflh[j],\
				par[j], bedrock_z[bedrock_index0][bedrock_index1]]
				unlabeled_dataset.append(sample_info)

print(len(unlabeled_dataset))

unlabeled_df = pd.DataFrame(unlabeled_dataset, columns=['Sample Date', 'Latitude', 'Longitude', \
	'aot_869', 'angstrom', 'Rrs_412', 'Rrs_443', 'Rrs_469', 'Rrs_488', 'Rrs_531', 'Rrs_547', 'Rrs_555', 'Rrs_645',\
	'Rrs_667', 'Rrs_678', 'chlor_a', 'chl_ocx', 'Kd_490', 'pic', 'poc', 'ipar', 'nflh', 'par', 'bedrock'])
unlabeled_df.to_pickle('unlabeledPixels/unlabeled_dataset.pkl')