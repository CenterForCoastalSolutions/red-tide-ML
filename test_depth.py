import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

load_folder = 'depth_stats'
filenames = ['angstrom_sums.npy', 'chlor_a_sums.npy', 'chl_ocx_sums.npy', 'Kd_490_sums.npy', 'poc_sums.npy', 'nflh_sums.npy', 'par_sums.npy', 'Rrs_443_sums.npy', 'Rrs_469_sums.npy', 'Rrs_488_sums.npy']

for filename in filenames:
	sums = np.load(load_folder+'/'+filename, allow_pickle='TRUE').item()

	keys = []
	#means = []
	#stds = []
	xs = []
	x2s = []
	ns = []

	for key in sums.keys():
		if(key[-1] == 'n'):
			key_strip = key[:-2]
			keys.append(int(key_strip))
			#means.append(sums[key_strip+'_x']/sums[key_strip+'_n'])
			#stds.append(math.sqrt((sums[key_strip+'_x2']/sums[key_strip+'_n'])-(means[-1]**2)))
			xs.append(sums[key_strip+'_x'])
			x2s.append(sums[key_strip+'_x2'])
			ns.append(sums[key_strip+'_n'])

	keys = np.array(keys)
	#means = np.array(means)
	#stds = np.array(stds)
	xs = np.array(xs)
	x2s = np.array(x2s)
	ns = np.array(ns)

	sort_inds = np.argsort(keys)

	keys = keys[sort_inds]
	#means = means[sort_inds]
	#stds = stds[sort_inds]
	xs = xs[sort_inds]
	x2s = x2s[sort_inds]
	ns = ns[sort_inds]

	means = np.zeros_like(xs)
	stds = np.zeros_like(xs)

	for i in range(len(xs)):
		total_x = xs[i]
		total_x2 = x2s[i]
		total_n = ns[i]
		# Included neighboring depths to smooth result
		if(i > 0):
			total_x += xs[i-1]
			total_x2 += x2s[i-1]
			total_n += ns[i-1]
		if((i+1) < len(xs)):
			total_x += xs[i+1]
			total_x2 += x2s[i+1]
			total_n += ns[i+1]
		means[i] = total_x/total_n
		stds[i] = math.sqrt((total_x2/total_n)-(means[i]**2))

	fill_xs = np.concatenate((keys, np.flip(keys)))
	fill_ys = np.concatenate((means+stds, np.flip(means-stds)))

	plt.figure(dpi=500)
	plt.plot(keys, means, 'b')
	plt.fill(fill_xs, fill_ys, alpha=0.3, facecolor='b')
	plt.xlim(-100, 2)
	#plt.ylim(0, 6)
	plt.xlabel('Bedrock Depth')
	plt.title(filename)
	plt.savefig('depth_stat_plots/'+filename[:-4]+'.png', bbox_inches='tight')

	plt.figure(dpi=500)
	plt.plot(keys, ns[sort_inds], 'b')
	plt.xlim(-100, 2)
	plt.ylim(0, 4000000)
	plt.xlabel('Bedrock Depth')
	plt.title(filename)
	plt.savefig('depth_stat_plots/'+filename[:-4]+'_n.png', bbox_inches='tight')

	np.save('depth_stats/'+filename[:-4]+'_keys.npy', keys)
	np.save('depth_stats/'+filename[:-4]+'_means.npy', means)
	np.save('depth_stats/'+filename[:-4]+'_stds.npy', stds)