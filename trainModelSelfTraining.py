import pandas as pd
import sys
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler
import time
import math
import copy
from random import sample
from model import *
from model_w_dropout import *
from dataset import *
from utils import *
from tqdm import tqdm
from convertFeaturesByDepth import *
from convertFeaturesByPosition import *
import datetime as dt
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix
import json
from configparser import ConfigParser
import matplotlib.pyplot as plt

def train(num_features, num_classes, learning_rate, validationDataloader, nonValidationDataloader):
	predictor = Predictor(num_features, num_classes).cuda()
	optimizer = optim.Adam(predictor.parameters(), lr=learning_rate)

	bestValPredictor = Predictor(num_features, num_classes).cuda()

	validationLosses = np.zeros((numEpochs, 1))
	nonValidationLosses = np.zeros((numEpochs, 1))

	bestValidationLoss = math.inf
	epochsWithoutImprovement = 0

	for i in range(numEpochs):
		predictor.eval()
		validationEpochLoss = 0
		with torch.no_grad():
			for mini_batch_data, mini_batch_labels in validationDataloader:
				output = predictor(mini_batch_data)
				miniBatchLoss = loss(output, mini_batch_labels)
				validationEpochLoss += miniBatchLoss.item()

		predictor.train()
		nonValidationEpochLoss = 0
		for mini_batch_data, mini_batch_labels in nonValidationDataloader:
			optimizer.zero_grad()
			output = predictor(mini_batch_data)
			miniBatchLoss = loss(output, mini_batch_labels)
			miniBatchLoss.backward()
			nonValidationEpochLoss += miniBatchLoss.item()
			optimizer.step()
		if(i%10==0):
			print('Epoch: {}, Non Validation Loss: {}, Validation Loss: {}'.format(i, nonValidationEpochLoss, validationEpochLoss))

		if(validationEpochLoss < bestValidationLoss):
			bestValidationLoss = validationEpochLoss
			epochsWithoutImprovement = 0
			bestValPredictor = copy.deepcopy(predictor)
		else:
			epochsWithoutImprovement += 1

		if(epochsWithoutImprovement > 50):
			print('Epoch {}, stopping early due to lack of improvement'.format(i))
			break

	return bestValPredictor


def knnSearch(numSamples, trainDates, trainLats, trainLons, searchDates, searchLats, searchLons):
	nn_classes = np.zeros((numSamples))
	knn_concs = np.zeros((numSamples))
	nearest_sample_dist = np.zeros((numSamples))
	#Do some nearest neighbor thing with the last week's samples
	for i in range(len(searchDates)):
		if(i%1000 == 0):
			print('Calculating KNN {}/{}'.format(i, len(searchDates)))

		searchdate = searchDates[i]
		weekbefore = searchdate - dt.timedelta(days=3)
		twoweeksbefore = searchdate - dt.timedelta(days=10)
		mask = (trainDates['Sample Date'] > twoweeksbefore) & (trainDates['Sample Date'] <= weekbefore)
		week_prior_inds = trainDates[mask].index.values

		if(week_prior_inds.size):
			physicalDistance = 100*np.sqrt((trainLats[week_prior_inds]-searchLats[i])**2 + (trainLons[week_prior_inds]-searchLons[i])**2)
			daysBack = (searchdate - trainDates['Sample Date'][week_prior_inds]).astype('timedelta64[D]').values
			totalDistance = physicalDistance + beta*daysBack
			inverseDistance = 1/totalDistance
			NN_weights = inverseDistance/np.sum(inverseDistance)
			
			idx = find_nearest_latlon(trainLats[week_prior_inds], trainLons[week_prior_inds], searchLats[i], searchLons[i])

			closestConc = df_concs[week_prior_inds][idx]
			knn_concs[i] = np.sum(NN_weights*df_concs_log[week_prior_inds])
			nearest_sample_dist[i] = np.min(physicalDistance)
			if(closestConc < 100000):
				nn_classes[i] = 0
			else:
				nn_classes[i] = 1
		else:
			nn_classes[i] = 0

	return nn_classes, knn_concs

np.set_printoptions(threshold=sys.maxsize)

configfilename = 'date_train_test_depth_norm_w_knn'

config = ConfigParser()
config.read('configfiles/'+configfilename+'.ini')

numEpochs = config.getint('main', 'numEpochs')
learning_rate = config.getfloat('main', 'learning_rate')
mb_size = config.getint('main', 'mb_size')
num_classes = config.getint('main', 'num_classes')
randomseeds = json.loads(config.get('main', 'randomseeds'))
normalization = config.getint('main', 'normalization')
traintest_split = config.getint('main', 'traintest_split')
use_nn_feature = config.getint('main', 'use_nn_feature')
balance_train = config.getint('main', 'balance_train')

loss = nn.BCELoss()

paired_df = pd.read_pickle('paired_dataset.pkl')

#features_to_use=['Sample Date', 'Latitude', 'aot_869', 'angstrom', 'Rrs_412', 'Rrs_443', 'Rrs_469', 'Rrs_488',\
#	'Rrs_531', 'Rrs_547', 'Rrs_555', 'Rrs_645',\
#	'Rrs_667', 'Rrs_678', 'chlor_a', 'chl_ocx', 'Kd_490', 'poc', 'par', 'ipar', 'nflh', 'Red Tide Concentration']
#features_to_use=['Sample Date', 'Latitude', 'Longitude', 'aot_869', 'par', 'ipar', 'angstrom', 'chlor_a', 'chl_ocx', 'Kd_490', 'poc', 'nflh', 'bedrock', 'Red Tide Concentration', 'Rrs_443', 'Rrs_555']
#features_to_use=['Sample Date', 'Latitude', 'Longitude', 'angstrom', 'chlor_a', 'chl_ocx', 'Kd_490', 'poc', 'nflh', 'bedrock', 'Red Tide Concentration', 'Rrs_443', 'Rrs_555']
features_to_use=['Sample Date', 'Latitude', 'Longitude', 'par', 'Kd_490', 'chlor_a', 'Rrs_443', 'Rrs_469', 'Rrs_488', 'nflh', 'bedrock', 'Red Tide Concentration']

paired_df = paired_df[features_to_use]

#Remove samples with NaN values
paired_df = paired_df.dropna()

red_tide = paired_df['Red Tide Concentration'].to_numpy().copy()
dates = paired_df['Sample Date'].to_numpy().copy()
latitudes = paired_df['Latitude'].to_numpy().copy()
longitudes = paired_df['Longitude'].to_numpy().copy()

features = paired_df[features_to_use[1:-1]]
features = np.array(features.values)


if(normalization == 0):
	features = features[:, 2:-1]
elif(normalization == 1):
	features = convertFeaturesByDepth(features[:, 2:], features_to_use[3:-2])
elif(normalization == 2):
	features = convertFeaturesByPosition(features[:, :-1], features_to_use[3:-2])

concentrations = red_tide
classes = np.zeros((concentrations.shape[0], 1))

for i in range(len(classes)):
	if(concentrations[i] < 100000):
		classes[i] = 0
	else:
		classes[i] = 1

beta = 1

# 0 = No nn features, 1 = nn, 2 = weighted knn
if(use_nn_feature == 1 or use_nn_feature == 2 or use_nn_feature == 3):
	file_path = 'PinellasMonroeCoKareniabrevis 2010-2020.06.12.xlsx'

	df = pd.read_excel(file_path, engine='openpyxl')
	df_dates = df['Sample Date']
	df_lats = df['Latitude'].to_numpy()
	df_lons = df['Longitude'].to_numpy()
	df_concs = df['Karenia brevis abundance (cells/L)'].to_numpy()

	df_conc_classes = np.zeros_like(df_concs)
	for i in range(len(df_conc_classes)):
		if(df_concs[i] < 100000):
			df_conc_classes[i] = 0
		else:
			df_conc_classes[i] = 1

	#Balance classes by number of samples
	values, counts = np.unique(df_conc_classes, return_counts=True)
	values = values[0:num_classes]
	counts = counts[0:num_classes]
	pointsPerClass = np.min(counts)
	reducedInds = np.array([])
	for i in range(num_classes):
		if(i==0):
			class_inds = np.where(df_concs < 100000)[0]
		else:
			class_inds = np.where(df_concs >= 100000)[0]
		reducedInds = np.append(reducedInds, class_inds[np.random.choice(class_inds.shape[0], pointsPerClass)])


	if(balance_train == 0):
		##### Don't balance data by classes
		reducedInds = np.array(range(len(df_concs)))



	reducedInds = reducedInds.astype(int)
	df_dates = df_dates[reducedInds]
	df_dates = df_dates.reset_index()
	df_lats = df_lats[reducedInds]
	df_lons = df_lons[reducedInds]
	df_concs = df_concs[reducedInds]
	df_concs_log = np.log10(df_concs)/np.max(np.log10(df_concs))
	df_concs_log[np.isinf(df_concs_log)] = 0

	dataDates = pd.DatetimeIndex(dates)
	
	nn_classes, knn_concs = knnSearch(features.shape[0], df_dates, df_lats, df_lons, dataDates, latitudes, longitudes)

	if(use_nn_feature == 1):
		ensure_folder('saved_model_info/'+configfilename)
		np.save('saved_model_info/'+configfilename+'/nn_classes.npy', nn_classes)
		original_features = np.copy(features)
		features = np.concatenate((features, np.expand_dims(nn_classes, axis=1)), axis=1)
	if(use_nn_feature == 2):
		ensure_folder('saved_model_info/'+configfilename)
		np.save('saved_model_info/'+configfilename+'/knn_concs.npy', knn_concs)
		original_features = np.copy(features)
		features = np.concatenate((features, np.expand_dims(knn_concs, axis=1)), axis=1)
	if(use_nn_feature == 3):
		ensure_folder('saved_model_info/'+configfilename)
		np.save('saved_model_info/'+configfilename+'/knn_concs.npy', knn_concs)
		original_features = np.copy(features)
		features = np.concatenate((features, np.expand_dims(knn_concs, axis=1)), axis=1)
		np.save('saved_model_info/'+configfilename+'/nearest_sample_dist.npy', nearest_sample_dist)



# Self-training
unlabeled_df = pd.read_pickle('unlabeledPixels/unlabeled_dataset.pkl')

unlabeled_df = unlabeled_df[features_to_use[0:-1]]

#Remove samples with NaN values
unlabeled_df = unlabeled_df.dropna()

unlabeled_features = unlabeled_df[features_to_use[1:-1]]
unlabeled_features = np.array(unlabeled_features.values)

#Only use 1m samples
unlabeled_features_sample = np.random.permutation(unlabeled_features.shape[0])
unlabeled_features = unlabeled_features[unlabeled_features_sample[0:1000000], :]

unlabeled_dates = unlabeled_df['Sample Date'].to_numpy().copy()
unlabeled_dates = unlabeled_dates[unlabeled_features_sample[0:1000000]]
unlabeled_latitudes = unlabeled_df['Latitude'].to_numpy().copy()
unlabeled_latitudes = unlabeled_latitudes[unlabeled_features_sample[0:1000000]]
unlabeled_longitudes = unlabeled_df['Longitude'].to_numpy().copy()
unlabeled_longitudes = unlabeled_longitudes[unlabeled_features_sample[0:1000000]]

if(normalization == 0):
	unlabeled_features = unlabeled_features[:, 2:-1]
elif(normalization == 1):
	unlabeled_features = convertFeaturesByDepth(unlabeled_features[:, 2:], features_to_use[3:-2])
elif(normalization == 2):
	unlabeled_features = convertFeaturesByPosition(unlabeled_features[:, :-1], features_to_use[3:-2])

unlabeled_dataDates = pd.DatetimeIndex(unlabeled_dates).tz_localize(None)

nn_classes, knn_concs = knnSearch(unlabeled_features.shape[0], df_dates, df_lats, df_lons, unlabeled_dataDates, unlabeled_latitudes, unlabeled_longitudes)

if(use_nn_feature == 1):
	unlabeled_features = np.concatenate((unlabeled_features, np.expand_dims(nn_classes, axis=1)), axis=1)
if(use_nn_feature == 2):
	unlabeled_features = np.concatenate((unlabeled_features, np.expand_dims(knn_concs, axis=1)), axis=1)
if(use_nn_feature == 3):
	unlabeled_features = np.concatenate((unlabeled_features, np.expand_dims(knn_concs, axis=1)), axis=1)

num_self_training_iters = 100


for model_number in range(len(randomseeds)):
	starttime = time.time()

	# Set up random seeds for reproducability
	torch.manual_seed(randomseeds[model_number])
	np.random.seed(randomseeds[model_number])

	#Balance classes by number of samples
	values, counts = np.unique(classes, return_counts=True)
	values = values[0:num_classes]
	counts = counts[0:num_classes]
	pointsPerClass = np.min(counts)
	reducedInds = np.array([])
	for i in range(num_classes):
		class_inds = np.where(classes == i)[0]
		reducedInds = np.append(reducedInds, class_inds[np.random.choice(class_inds.shape[0], pointsPerClass)])


	if(balance_train == 0):
		##### Don't balance data by classes
		reducedInds = np.array(range(len(classes)))



	reducedInds = reducedInds.astype(int)

	usedClasses = classes[reducedInds]

	featuresTensor = torch.tensor(features)

	reducedFeaturesTensor = featuresTensor[reducedInds, :]

	usedDates = dates[reducedInds]
	usedLatitudes = latitudes[reducedInds]

	years_in_data = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,\
					 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
	#years_in_data = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,\
	#				 2010, 2011, 2012]

	if(traintest_split == 0):
		trainInds = sample(range(reducedFeaturesTensor.shape[0]), int(0.8*reducedFeaturesTensor.shape[0]))
		testInds = list(set(range(reducedFeaturesTensor.shape[0]))-set(trainInds))
	elif(traintest_split == 1):
		#Select the years to test on
		years_to_test = np.random.choice(years_in_data, size=2, replace=False)

		trainInds = np.logical_and(np.logical_or(usedDates < np.datetime64(str(years_to_test[0])+'-01-01'), usedDates >= np.datetime64(str(years_to_test[0]+1)+'-01-01')), np.logical_or(usedDates < np.datetime64(str(years_to_test[1])+'-01-01'), usedDates >= np.datetime64(str(years_to_test[1]+1)+'-01-01')))
		testInds = np.logical_or(np.logical_and(usedDates >= np.datetime64(str(years_to_test[0])+'-01-01'), usedDates < np.datetime64(str(years_to_test[0]+1)+'-01-01')), np.logical_and(usedDates >= np.datetime64(str(years_to_test[1])+'-01-01'), usedDates < np.datetime64(str(years_to_test[1]+1)+'-01-01')))
	elif(traintest_split == 2):
		trainInds = np.logical_or(usedLatitudes >= 27, usedLatitudes < 26.5)
		testInds = np.logical_and(usedLatitudes < 27, usedLatitudes >= 26.5)

	trainSet = reducedFeaturesTensor[trainInds, :].float().cuda()
	testSet = reducedFeaturesTensor[testInds, :].float().cuda()

	trainClasses = usedClasses[trainInds]

	trainClasses = trainClasses.astype(int)

	trainTargets = np.zeros((trainClasses.shape[0], num_classes))
	for i in range(len(trainClasses)):
		trainTargets[i, trainClasses[i]] = 1

	testClasses = usedClasses[testInds]

	testClasses = testClasses.astype(int)

	testTargets = np.zeros((testClasses.shape[0], num_classes))
	for i in range(len(testClasses)):
		testTargets[i, testClasses[i]] = 1


	# Select 20% of data points for validation
	nonValidationInds = sample(range(trainSet.shape[0]), int(0.8*trainSet.shape[0]))
	validationInds = list(set(range(trainSet.shape[0]))-set(nonValidationInds))

	nonValidationData = trainSet[nonValidationInds, :]
	validationData = trainSet[validationInds, :]
	nonValidationTargets = trainTargets[nonValidationInds]
	validationTargets = trainTargets[validationInds]


	nonValidationTargets = torch.Tensor(nonValidationTargets).float().cuda()
	validationTargets = torch.Tensor(validationTargets).float().cuda()
	testTargets = torch.Tensor(testTargets).float().cuda()


	#Weight sampling by class
	#posFrac = np.sum(trainTargets[:,1])/trainTargets.shape[0]
	#negFrac = 1-posFrac
	#nonValidationWeights = torch.zeros((nonValidationTargets.shape[0], 1))
	#for i in range(nonValidationWeights.shape[0]):
	#	if(nonValidationTargets[i, 0] == 1):
	#		nonValidationWeights[i] = posFrac
	#	else:
	#		nonValidationWeights[i] = negFrac
	#validationWeights = torch.zeros((validationTargets.shape[0], 1))
	#for i in range(validationWeights.shape[0]):
	#	if(validationTargets[i, 0] == 1):
	#		validationWeights[i] = posFrac
	#	else:
	#		validationWeights[i] = negFrac
	#nonValidationWeights = nonValidationWeights.float().cuda()
	#validationWeights = validationWeights.float().cuda()
	#nonValidationSampler = torch.utils.data.sampler.WeightedRandomSampler(nonValidationWeights, mb_size)
	#validationSampler = torch.utils.data.sampler.WeightedRandomSampler(validationWeights, mb_size)


	nonValidationDataset = RedTideDataset(nonValidationData, nonValidationTargets)
	nonValidationDataloader = DataLoader(nonValidationDataset, batch_size=mb_size, shuffle=True)

	validationDataset = RedTideDataset(validationData, validationTargets)
	validationDataloader = DataLoader(validationDataset, batch_size=mb_size, shuffle=True)

	bestValPredictor = train(trainSet.shape[1], num_classes, learning_rate, validationDataloader, nonValidationDataloader)

	ensure_folder('saved_model_info/'+configfilename)

	torch.save(bestValPredictor.state_dict(), 'saved_model_info/'+configfilename+'/predictor{}.pt'.format(model_number))
	np.save('saved_model_info/'+configfilename+'/reducedInds{}.npy'.format(model_number), reducedInds)
	np.save('saved_model_info/'+configfilename+'/testInds{}.npy'.format(model_number), testInds)

	endtime = time.time()

	print('Model training time: {}'.format(endtime-starttime))

	unlabeled_features_trimmed = np.copy(unlabeled_features)
	unlabeledFeaturesTensor = torch.tensor(unlabeled_features_trimmed).float().cuda()

	for i in range(num_self_training_iters):

		# Predict with unlabeled data

		unlabeledPredictions = np.zeros((unlabeledFeaturesTensor.shape[0] , 2))

		bestValPredictor.eval()
		output = bestValPredictor(unlabeledFeaturesTensor)
		unlabeledPredictions = output.detach().cpu().numpy()

		# Take most confident predictions (balance by class?)

		class_winners = unlabeledPredictions[:, 0] - unlabeledPredictions[:, 1]
		class_winners_inds = np.argsort(class_winners)

		#class0MostConfident = class_winners_inds[-90:]
		#class1MostConfident = class_winners_inds[:10]
		class0_pred = np.where(class_winners > 0.3)[0]
		class1_pred = np.where(class_winners < -0.3)[0]
		knn_feature_class0_pred = np.argsort(unlabeled_features_trimmed[class0_pred, -1])
		knn_feature_class1_pred = np.argsort(unlabeled_features_trimmed[class1_pred, -1])
		probablyShouldBeClass1 = class0_pred[knn_feature_class0_pred[-10:]]
		probablyShouldBeClass0 = class1_pred[knn_feature_class1_pred[:90]]
		#plt.figure(dpi=500)
		#plt.hist(unlabeled_features_trimmed[class0_pred, -1], bins=50, alpha=0.5, density=1, color='c', label='Predicted Negatives')
		#plt.hist(unlabeled_features_trimmed[class1_pred, -1], bins=50, alpha=0.5, density=1, color='m', label='Predicted Positives')
		#plt.legend()
		#plt.savefig('unlabeledFeatureInfo/test.png')
				
		nonValidationDataWUnlabeled = torch.cat((nonValidationData, unlabeledFeaturesTensor[probablyShouldBeClass0, :]), 0)
		class0UnlabeledTargets = torch.zeros((probablyShouldBeClass0.shape[0], num_classes)).float().cuda()
		class0UnlabeledTargets[:, 0] = 1
		nonValidationTargetsWUnlabeled = torch.cat((nonValidationTargets, class0UnlabeledTargets))

		nonValidationDataWUnlabeled = torch.cat((nonValidationDataWUnlabeled, unlabeledFeaturesTensor[probablyShouldBeClass1, :]), 0)
		class1UnlabeledTargets = torch.zeros((probablyShouldBeClass1.shape[0], num_classes)).float().cuda()
		class1UnlabeledTargets[:, 1] = 1
		nonValidationTargetsWUnlabeled = torch.cat((nonValidationTargetsWUnlabeled, class1UnlabeledTargets))

		np.save('nonValidationDataWUnlabeled.npy', nonValidationDataWUnlabeled.cpu().numpy())
		np.save('nonValidationTargetsWUnlabeled.npy', nonValidationTargetsWUnlabeled.cpu().numpy())

		nonValidationDatasetWUnlabeled = RedTideDataset(nonValidationDataWUnlabeled, nonValidationTargetsWUnlabeled)
		nonValidationDataloaderWUnlabeled = DataLoader(nonValidationDatasetWUnlabeled, batch_size=mb_size, shuffle=True)

		# Remove pseudo-labeled samples from the unlabeled set so they aren't selected again next time

		indsToRemove = np.concatenate((probablyShouldBeClass0, probablyShouldBeClass1))
		unlabeled_features_trimmed = np.delete(unlabeled_features_trimmed, indsToRemove, axis=0)
		unlabeledFeaturesTensor = torch.tensor(unlabeled_features_trimmed).float().cuda()

		# Train again 

		starttime = time.time()

		bestValPredictor = train(trainSet.shape[1], num_classes, learning_rate, validationDataloader, nonValidationDataloaderWUnlabeled)

		ensure_folder('saved_model_info/'+configfilename+'/SelfTraining/{}Iteration'.format(i+1))

		torch.save(bestValPredictor.state_dict(), 'saved_model_info/'+configfilename+'/SelfTraining/{}Iteration/predictor{}.pt'.format(i+1, model_number))

		endtime = time.time()

		print('Model training time: {}'.format(endtime-starttime))