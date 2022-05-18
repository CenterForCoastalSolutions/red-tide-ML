import numpy as np
import netCDF4
import math
import xarray as xr
import matplotlib.pyplot as plt


track_starts_per_day = np.load('track_starts_per_day.npy')
track_starts_lats = np.load('track_starts_lats.npy')
track_starts_lons = np.load('track_starts_lons.npy')

num_points_initial = 10000
num_points_per_day = 1000

chlor_variogram_info = []

initial_perm = np.random.permutation(track_starts_per_day.shape[0])
for i in range(num_points_initial):
	if(i % 100 == 0):
		print('Processing {}/{}'.format(i, num_points_initial))

	if(abs(track_starts_lats[initial_perm[i]]) > 180 or abs(track_starts_lons[initial_perm[i]]) > 180):
		continue

	for j in range(track_starts_per_day.shape[1]):
		# Per day, compare to some number of other points
		day_perm = np.random.permutation(track_starts_per_day.shape[0])
		for k in range(num_points_per_day):
			if(abs(track_starts_lats[day_perm[k]]) > 180 or abs(track_starts_lons[day_perm[k]]) > 180):
				continue

			disp = math.sqrt((track_starts_lats[initial_perm[i]]-track_starts_lats[day_perm[k]])**2 + (track_starts_lons[initial_perm[i]]-track_starts_lons[day_perm[k]])**2)
			time_diff = j
			chlor_diff = abs(track_starts_per_day[initial_perm[i], 0]-track_starts_per_day[day_perm[k], j])
			chlor_variogram_info.append([time_diff, disp, chlor_diff])

chlor_variogram_info = np.array(chlor_variogram_info)
np.save('chlor_variogram_info.npy', chlor_variogram_info)
asdf


data_xarray_chlor_a = np.load('data_xarray_chlor_a.npy')

data_netcdf4 = netCDF4.Dataset('/run/media/rfick/UF10/HYCOM/hycom_correlations.nc', mode='r')
data_xarray = xr.open_dataset(xr.backends.NetCDF4DataStore(data_netcdf4))

chlor_vs_track = []

for i in range(data_xarray_chlor_a.shape[0]):
	if(i % 100 == 0):
		print('Processing {}/{}'.format(i, data_xarray_chlor_a.shape[0]))

	timearray = np.array(data_xarray['time'].values[i, :])
	latarray = np.array(data_xarray['lat'].values[i, :])
	lonarray = np.array(data_xarray['lon'].values[i, :])

	#Find last valid index
	lst = [j for j in timearray if not np.isnat(j)]
	
	total_track_length = 0
	for j in range(1, len(lst)):
		chlor_diff = abs(data_xarray_chlor_a[i, 0]-data_xarray_chlor_a[i, j])
		if(abs(latarray[j]) > 180 or abs(lonarray[j]) > 180):
			continue
		else:
			dist_diff = math.sqrt((latarray[j-1]-latarray[j])**2 + (lonarray[j-1]-lonarray[j])**2)
		total_track_length += dist_diff
		track_disp = math.sqrt((latarray[0]-latarray[j])**2 + (lonarray[0]-lonarray[j])**2)
		time_diff = abs(timearray[0]-timearray[j])/np.timedelta64(1, 'D')
		chlor_vs_track.append([total_track_length, time_diff, track_disp, chlor_diff])

chlor_vs_track = np.array(chlor_vs_track)
np.save('chlor_vs_track.npy', chlor_vs_track)
asdf

path_diffs = []

data_perm = np.random.permutation(data_xarray['time'].shape[0])

for i in range(100000):
	data_ind = data_perm[i]

	timearray = np.array(data_xarray['time'].values[data_ind, :])
	latarray = np.array(data_xarray['lat'].values[data_ind, :])
	lonarray = np.array(data_xarray['lon'].values[data_ind, :])

	#Find last valid index
	lst = [j for j in timearray if not np.isnat(j)]
	lst_ind = len(lst)-1

	chlor_diff = abs(data_xarray_chlor_a[data_ind, 0]-data_xarray_chlor_a[data_ind, lst_ind])
	time_diff = abs(timearray[0]-timearray[lst_ind])/np.timedelta64(1, 'D')
	dist_diff = math.sqrt((latarray[0]-latarray[lst_ind])**2 + (lonarray[0]-lonarray[lst_ind])**2)

	path_diffs.append([dist_diff, time_diff, chlor_diff])

path_diffs = np.array(path_diffs)

fig = plt.figure(dpi=500)
ax = fig.add_subplot(projection='3d')
ax.scatter(np.squeeze(path_diffs[:, 0]), np.squeeze(path_diffs[:, 1]), np.squeeze(path_diffs[:, 2]))
ax.set_xlabel('Physical Distance')
ax.set_ylabel('Time Difference')
ax.set_zlabel('Chlorophyll Difference')
plt.savefig('test.png')