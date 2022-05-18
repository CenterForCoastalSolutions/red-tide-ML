import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

latLonToKM = 111

chlor_variogram_info = np.load('chlor_variogram_info.npy')

hist, bin_edges_time = np.histogram(chlor_variogram_info[:, 0], bins=29)
indices = np.digitize(chlor_variogram_info[:, 0], bin_edges_time)

meanDiffs = np.zeros((len(bin_edges_time), 1))
stdDiffs = np.zeros((len(bin_edges_time), 1))

for i in range(len(bin_edges_time)):
	thisBin = np.where(indices == i+1)[0]
	meanDiffs[i] = np.mean(chlor_variogram_info[thisBin, 2])
	stdDiffs[i] = np.std(chlor_variogram_info[thisBin, 2])

x_vals = np.concatenate((bin_edges_time, np.flip(bin_edges_time)))
y_vals = np.concatenate((meanDiffs+0.1*stdDiffs, np.flip(meanDiffs)-0.1*np.flip(stdDiffs)))

plt.figure(dpi=500)
plt.fill(x_vals, y_vals, alpha=0.5)
plt.plot(bin_edges_time, meanDiffs)
plt.ylim(-0.1, 2.5)
plt.xlabel('Isotropic Time (Days)')
plt.ylabel('Mean Chlorophyll-a Difference')
plt.title('Chlorophyll-a vs Isotropic Time')
plt.savefig('chlor_variogram_time.png', bbox_inches='tight')

hist, bin_edges_disp = np.histogram(chlor_variogram_info[:, 1], bins=50)
indices = np.digitize(chlor_variogram_info[:, 1], bin_edges_disp)

meanDiffs = np.zeros((len(bin_edges_disp), 1))
stdDiffs = np.zeros((len(bin_edges_disp), 1))

for i in range(len(bin_edges_disp)):
	thisBin = np.where(indices == i+1)[0]
	meanDiffs[i] = np.mean(chlor_variogram_info[thisBin, 2])
	stdDiffs[i] = np.std(chlor_variogram_info[thisBin, 2])

x_vals = np.concatenate((bin_edges_disp, np.flip(bin_edges_disp)))
y_vals = np.concatenate((meanDiffs+0.1*stdDiffs, np.flip(meanDiffs)-0.1*np.flip(stdDiffs)))

plt.figure(dpi=500)
plt.fill(latLonToKM*x_vals, y_vals, alpha=0.5)
plt.plot(latLonToKM*bin_edges_disp, meanDiffs)
plt.ylim(-0.1, 17)
plt.xlabel('Isotropic Displacement (km)')
plt.ylabel('Mean Chlorophyll-a Difference')
plt.title('Chlorophyll-a vs Isotropic Displacement')
plt.savefig('chlor_variogram_dist.png', bbox_inches='tight')


compute_variogram = False

if(compute_variogram == True):
	variogram_sums = np.zeros((len(bin_edges_time), len(bin_edges_disp)))
	variogram_counts = np.zeros((len(bin_edges_time), len(bin_edges_disp)))
	variogram_vals = np.zeros((len(bin_edges_time), len(bin_edges_disp)))
	for i in range(chlor_variogram_info.shape[0]):
		if(i%1000 == 0):
			print('Processing {}/{}'.format(i, chlor_variogram_info.shape[0]))
		#Find which bins this point goes into
		time = chlor_variogram_info[i, 0]
		disp = chlor_variogram_info[i, 1]

		time_bin = np.searchsorted(bin_edges_time, time)
		disp_bin = np.searchsorted(bin_edges_disp, disp)

		variogram_sums[time_bin, disp_bin] += chlor_variogram_info[i, 2]
		variogram_counts[time_bin, disp_bin] += 1

	np.divide(variogram_sums, variogram_counts, variogram_vals)

	np.save('variogram_vals.npy', variogram_vals)
else:
	variogram_vals = np.load('variogram_vals.npy')

variogram_xs = np.tile(bin_edges_time, (variogram_vals.shape[1], 1)).T
variogram_ys = np.tile(bin_edges_disp.T, (variogram_vals.shape[0], 1))

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
mappable = plt.cm.ScalarMappable(cmap=cm.coolwarm)
mappable.set_array(variogram_vals)
mappable.set_clim(0, 2)
surf = ax.plot_surface(variogram_xs, latLonToKM*variogram_ys, variogram_vals, cmap=mappable.cmap, norm=mappable.norm, linewidth=0, antialiased=False)
#surf = ax.plot_surface(variogram_xs, latLonToKM*variogram_ys, variogram_vals, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(0, 2)
#plt.clim(0, 2)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('Time (days)')
plt.ylabel('Displacement (km)')
plt.title('Chlorophyll-a Difference Isotropic')
plt.savefig('chlor_variogram_zoomed.png')





chlor_vs_track = np.load('chlor_vs_track.npy')

hist, bin_edges = np.histogram(chlor_vs_track[:, 0], bins=50)
indices = np.digitize(chlor_vs_track[:, 0], bin_edges)

meanDiffs = np.zeros((len(bin_edges), 1))
stdDiffs = np.zeros((len(bin_edges), 1))

for i in range(len(bin_edges)):
	thisBin = np.where(indices == i+1)[0]
	meanDiffs[i] = np.mean(chlor_vs_track[thisBin, 3])
	stdDiffs[i] = np.std(chlor_vs_track[thisBin, 3])

x_vals = np.concatenate((bin_edges, np.flip(bin_edges)))
y_vals = np.concatenate((meanDiffs+0.1*stdDiffs, np.flip(meanDiffs)-0.1*np.flip(stdDiffs)))

plt.figure(dpi=500)
plt.fill(latLonToKM*x_vals, y_vals, alpha=0.5)
plt.plot(latLonToKM*bin_edges, meanDiffs)
plt.xlabel('Track Length (km)')
plt.ylabel('Mean Chlorophyll-a Difference')
plt.title('Chlorophyll-a vs Track Length')
plt.savefig('chlor_vs_track.png', bbox_inches='tight')

hist2, bin_edges_time = np.histogram(chlor_vs_track[:, 1], bins=50)
indices2 = np.digitize(chlor_vs_track[:, 1], bin_edges_time)

meanDiffs2 = np.zeros((len(bin_edges_time), 1))
stdDiffs2 = np.zeros((len(bin_edges_time), 1))

for i in range(len(bin_edges_time)):
	thisBin = np.where(indices2 == i+1)[0]
	meanDiffs2[i] = np.mean(chlor_vs_track[thisBin, 3])
	stdDiffs2[i] = np.std(chlor_vs_track[thisBin, 3])

x_vals = np.concatenate((bin_edges_time, np.flip(bin_edges_time)))
y_vals = np.concatenate((meanDiffs2+0.1*stdDiffs2, np.flip(meanDiffs2)-0.1*np.flip(stdDiffs2)))

plt.figure(dpi=500)
plt.fill(x_vals, y_vals, alpha=0.5)
plt.plot(bin_edges_time, meanDiffs2)
plt.ylim(-0.1, 2.5)
plt.xlabel('Track Time (Days)')
plt.ylabel('Mean Chlorophyll-a Difference')
plt.title('Chlorophyll-a vs Track Time')
plt.savefig('chlor_vs_time.png', bbox_inches='tight')

hist3, bin_edges_disp = np.histogram(chlor_vs_track[:, 2], bins=50)
indices3 = np.digitize(chlor_vs_track[:, 2], bin_edges_disp)

meanDiffs3 = np.zeros((len(bin_edges_disp), 1))
stdDiffs3 = np.zeros((len(bin_edges_disp), 1))

for i in range(len(bin_edges_disp)):
	thisBin = np.where(indices3 == i+1)[0]
	meanDiffs3[i] = np.mean(chlor_vs_track[thisBin, 3])
	stdDiffs3[i] = np.std(chlor_vs_track[thisBin, 3])

x_vals = np.concatenate((bin_edges_disp, np.flip(bin_edges_disp)))
y_vals = np.concatenate((meanDiffs3+0.1*stdDiffs3, np.flip(meanDiffs3)-0.1*np.flip(stdDiffs3)))

plt.figure(dpi=500)
plt.fill(latLonToKM*x_vals, y_vals, alpha=0.5)
plt.plot(latLonToKM*bin_edges_disp, meanDiffs3)
plt.ylim(-0.1, 17)
plt.xlabel('Track Displacement (km)')
plt.ylabel('Mean Chlorophyll-a Difference')
plt.title('Chlorophyll-a vs Track Displacement')
plt.savefig('chlor_vs_track_disp.png', bbox_inches='tight')



compute_variogram = False

if(compute_variogram == True):
	variogram_track_sums = np.zeros((len(bin_edges_time), len(bin_edges_disp)))
	variogram_track_counts = np.zeros((len(bin_edges_time), len(bin_edges_disp)))
	variogram_track_vals = np.zeros((len(bin_edges_time), len(bin_edges_disp)))
	for i in range(chlor_vs_track.shape[0]):
		if(i%1000 == 0):
			print('Processing {}/{}'.format(i, chlor_vs_track.shape[0]))
		#Find which bins this point goes into
		time = chlor_vs_track[i, 1]
		disp = chlor_vs_track[i, 2]

		time_bin = np.searchsorted(bin_edges_time, time)
		disp_bin = np.searchsorted(bin_edges_disp, disp)

		variogram_track_sums[time_bin, disp_bin] += chlor_vs_track[i, 3]
		variogram_track_counts[time_bin, disp_bin] += 1

	np.divide(variogram_track_sums, variogram_track_counts, variogram_track_vals)

	np.save('variogram_track_vals.npy', variogram_track_vals)
else:
	variogram_track_vals = np.load('variogram_track_vals.npy')

variogram_track_vals[np.isnan(variogram_track_vals)] = 0

variogram_xs = np.tile(bin_edges_time, (variogram_track_vals.shape[1], 1)).T
variogram_ys = np.tile(bin_edges_disp.T, (variogram_track_vals.shape[0], 1))

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
mappable = plt.cm.ScalarMappable(cmap=cm.coolwarm)
mappable.set_array(variogram_vals)
mappable.set_clim(0, 2)
surf = ax.plot_surface(variogram_xs, latLonToKM*variogram_ys, variogram_track_vals, cmap=mappable.cmap, norm=mappable.norm, linewidth=0, antialiased=False)
#surf = ax.plot_surface(variogram_xs, latLonToKM*variogram_ys, variogram_track_vals, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(0, 2)
#plt.clim(0, 2)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('Time (days)')
plt.ylabel('Displacement (km)')
plt.title('Chlorophyll-a Difference Track')
plt.savefig('chlor_track_variogram_zoomed.png')