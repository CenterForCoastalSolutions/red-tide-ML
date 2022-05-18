import numpy as np
import matplotlib.pyplot as plt

chlor_a_pairs = np.load("/run/media/rfick/UF10/HYCOM/chlor_a_pairs.npy", allow_pickle=True)

unique_times = np.unique(chlor_a_pairs[:,0])

chlor_a_means = np.zeros_like(unique_times)

for i in range(len(unique_times)):
	time_inds = np.where(chlor_a_pairs[:,0] == unique_times[i])[0]
	chlor_a_means[i] = np.mean(chlor_a_pairs[time_inds, 1])

	if(unique_times[i] == 0):
		plt.figure(dpi=500)
		plt.hist(chlor_a_pairs[time_inds, 1], bins=50)
		non_zero_inds = np.where(chlor_a_pairs[time_inds, 1] != 0)[0]
		plt.title('Chlor-a Diffs at Time 0')
		plt.savefig('test.png')

print(chlor_a_means)

plt.figure(dpi=500)
plt.scatter(unique_times, chlor_a_means)
plt.ylim(0, 5)
plt.ylabel('chlor_a error')
plt.xlabel('Hours Apart')
plt.savefig('chlor_a_pairs.png')

chlor_a_sameloc = np.load("/run/media/rfick/UF10/HYCOM/chlor_a_sameloc.npy", allow_pickle=True)

unique_times = np.unique(chlor_a_sameloc[:,0])

chlor_a_means = np.zeros_like(unique_times)

for i in range(len(unique_times)):
	time_inds = np.where(chlor_a_sameloc[:,0] == unique_times[i])[0]
	chlor_a_means[i] = np.mean(chlor_a_sameloc[time_inds, 1])

print(chlor_a_means)

plt.figure(dpi=500)
plt.scatter(unique_times, chlor_a_means)
plt.ylim(0, 5)
plt.ylabel('chlor_a error')
plt.xlabel('Hours Apart')
plt.savefig('chlor_a_sameloc.png')