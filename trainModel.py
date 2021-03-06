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
from geopy.distance import geodesic
import datetime as dt
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix
import json
from configparser import ConfigParser
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

configfilename = 'date_train_test_depth_norm_w_knn_50k_hill'

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
detection_limit = config.getint('main', 'detection_limit')
use_hill_test = config.getint('main', 'use_hill_test')

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



#Filter out samples between 0 and the detection_limit
if(use_hill_test == 1):
	valid_inds = (red_tide == 0) | (red_tide > detection_limit)
	valid_inds = np.where(valid_inds == True)[0]

	red_tide = red_tide[valid_inds]
	dates = dates[valid_inds]
	latitudes = latitudes[valid_inds]
	longitudes = longitudes[valid_inds]
	features = features[valid_inds, :]




concentrations = red_tide
classes = np.zeros((concentrations.shape[0], 1))

for i in range(len(classes)):
	if(concentrations[i] < detection_limit):
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
		if(df_concs[i] < detection_limit):
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
			class_inds = np.where(df_concs < detection_limit)[0]
		else:
			class_inds = np.where(df_concs >= detection_limit)[0]
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
	nn_classes = np.zeros((features.shape[0]))
	knn_concs = np.zeros((features.shape[0]))
	nearest_sample_dist = np.zeros((features.shape[0]))
	#Do some nearest neighbor thing with the last week's samples
	for i in range(len(dataDates)):
		searchdate = dataDates[i]
		weekbefore = searchdate - dt.timedelta(days=3)
		twoweeksbefore = searchdate - dt.timedelta(days=10)
		mask = (df_dates['Sample Date'] > twoweeksbefore) & (df_dates['Sample Date'] <= weekbefore)
		week_prior_inds = df_dates[mask].index.values

		if(week_prior_inds.size):
			physicalDistance = np.zeros((len(week_prior_inds), 1))
			for j in range(len(week_prior_inds)):
				oldLocation = (df_lats[week_prior_inds][j], df_lons[week_prior_inds][j])
				currLocation = (latitudes[i], longitudes[i])
				physicalDistance[j] = geodesic(oldLocation, currLocation).km
			daysBack = (searchdate - df_dates['Sample Date'][week_prior_inds]).astype('timedelta64[D]').values
			totalDistance = physicalDistance + beta*daysBack
			inverseDistance = 1/totalDistance
			NN_weights = inverseDistance/np.sum(inverseDistance)

			idx = find_nearest_latlon(df_lats[week_prior_inds], df_lons[week_prior_inds], latitudes[i], longitudes[i])

			closestConc = df_concs[week_prior_inds][idx]
			knn_concs[i] = np.sum(NN_weights*df_concs_log[week_prior_inds])
			nearest_sample_dist[i] = np.min(physicalDistance)
			if(closestConc < detection_limit):
				nn_classes[i] = 0
			else:
				nn_classes[i] = 1
		else:
			nn_classes[i] = 0

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


	# Hill uses a split of 1755 positive examples and 1114 negative examples
	# Try to match that ratio to make the metrics match
	if(use_hill_test == 1):
		#Balance classes by number of samples
		values, counts = np.unique(df_conc_classes, return_counts=True)
		reducedInds = np.array([])
		class_inds = np.where(classes == 1)[0]
		reducedInds = np.append(reducedInds, class_inds)
		numPositives = len(class_inds)
		numNegativesToSelect = int(numPositives/(1755/1114))
		class_inds = np.where(classes == 0)[0]
		reducedInds = np.append(reducedInds, class_inds[np.random.choice(class_inds.shape[0], numNegativesToSelect)])
		reducedInds = reducedInds.astype(int)


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

	nonValidationDataset = RedTideDataset(nonValidationData, nonValidationTargets)
	nonValidationDataloader = DataLoader(nonValidationDataset, batch_size=mb_size, shuffle=True)

	validationDataset = RedTideDataset(validationData, validationTargets)
	validationDataloader = DataLoader(validationDataset, batch_size=mb_size, shuffle=True)

	predictor = Predictor(trainSet.shape[1], num_classes).cuda()
	optimizer = optim.Adam(predictor.parameters(), lr=learning_rate)

	bestValPredictor = Predictor(trainSet.shape[1], num_classes).cuda()

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
		validationLosses[i] = validationEpochLoss
		nonValidationLosses[i] = nonValidationEpochLoss

		if(validationEpochLoss < bestValidationLoss):
			bestValidationLoss = validationEpochLoss
			epochsWithoutImprovement = 0
			bestValPredictor = copy.deepcopy(predictor)
		else:
			epochsWithoutImprovement += 1

		if(epochsWithoutImprovement > 50):
			print('Epoch {}, stopping early due to lack of improvement'.format(i))
			break

	ensure_folder('saved_model_info/'+configfilename)

	plt.figure(dpi=500)
	plt.plot(validationLosses, 'b', label='Validation Loss')
	plt.plot(nonValidationLosses, 'r', label='Non-Validation Loss')
	plt.legend()
	plt.savefig('losses.png')

	torch.save(bestValPredictor.state_dict(), 'saved_model_info/'+configfilename+'/predictor{}.pt'.format(model_number))
	np.save('saved_model_info/'+configfilename+'/reducedInds{}.npy'.format(model_number), reducedInds)
	np.save('saved_model_info/'+configfilename+'/testInds{}.npy'.format(model_number), testInds)

	endtime = time.time()

	print('Model training time: {}'.format(endtime-starttime))