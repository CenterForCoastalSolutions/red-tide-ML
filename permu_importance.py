import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import datetime as dt
import traceback
from random import sample
from model import *
from model_w_dropout import *
from dataset import *
from utils import *
from convertROC import *
from geopy.distance import geodesic
#from hycom_dist_parcels import *
from file_search_netCDF import *
from convertFeaturesByDepth import *
from convertFeaturesByPosition import *
from detectors.SotoEtAlDetector import *
from detectors.AminEtAlRBDDetector import *
from detectors.AminEtAlRBDKBBIDetector import *
from detectors.StumpfEtAlDetector import *
from detectors.Cannizzaro2008EtAlDetector import *
from detectors.Cannizzaro2009EtAlDetector import *
from detectors.ShehhiEtAlDetector import *
from detectors.Tomlinson2009EtAlDetector import *
from detectors.LouEtAlDetector import *
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
import json
from configparser import ConfigParser
import matplotlib.pyplot as plt

def addNewModelResultsThreshold(output, testClasses, fprs, tprs):
	false_positives = 0
	true_positives = 0
	total_negatives = 0
	total_positives = 0

	for i in range(len(testClasses)):
		if(testClasses[i] == 0):
			if(output[i] != 0):
				false_positives += 1
			total_negatives += 1
		else:
			if(output[i] == 1):
				true_positives += 1
			total_positives += 1

	fpr = false_positives/total_negatives
	tpr = true_positives/total_positives
	fprs.append(fpr)
	tprs.append(tpr)
	return fprs, tprs

def addNewModelResultsROC(output, testClasses, refFpr, tprs):
	fpr, tpr, thresholds = roc_curve(testClasses, output)
	if(model_number == 0):
		refFpr = fpr
		tprs = tpr
		tprs = np.expand_dims(tprs, axis=1)
	else:
		refTpr = convertROC(fpr, tpr, refFpr)
		refTpr = np.expand_dims(refTpr, axis=1)
		tprs = np.concatenate((tprs, refTpr), axis=1)
	return refFpr, tprs

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
df_classes = df_conc_classes[reducedInds]

paired_df = pd.read_pickle('paired_dataset.pkl')

#features_to_use=['Sample Date', 'Latitude', 'aot_869', 'angstrom', 'Rrs_412', 'Rrs_443', 'Rrs_469', 'Rrs_488',\
#	'Rrs_531', 'Rrs_547', 'Rrs_555', 'Rrs_645',\
#	'Rrs_667', 'Rrs_678', 'chlor_a', 'chl_ocx', 'Kd_490', 'poc', 'par', 'ipar', 'nflh', 'Red Tide Concentration']
#features_to_use=['Sample Date', 'Latitude', 'Longitude', 'aot_869', 'par', 'ipar', 'angstrom', 'chlor_a', 'chl_ocx', 'Kd_490', 'poc', 'nflh', 'bedrock', 'Red Tide Concentration', 'Rrs_443', 'Rrs_555', 'Rrs_667', 'Rrs_678']
#features_to_use=['Sample Date', 'Latitude', 'Longitude', 'angstrom', 'chlor_a', 'chl_ocx', 'Kd_490', 'poc', 'nflh', 'bedrock', 'Red Tide Concentration', 'Rrs_443', 'Rrs_555', 'Rrs_667', 'Rrs_678', 'Rrs_469', 'Rrs_488', 'Rrs_531']
features_to_use=['Sample Date', 'Latitude', 'Longitude', 'par', 'Kd_490', 'chlor_a', 'Rrs_443', 'Rrs_469', 'Rrs_488', 'nflh', 'bedrock', 'Red Tide Concentration', 'Rrs_555', 'Rrs_667', 'Rrs_678', 'Rrs_531', 'chl_ocx']

paired_df = paired_df[features_to_use]

#Remove samples with NaN values
paired_df = paired_df.dropna()

red_tide = paired_df['Red Tide Concentration'].to_numpy().copy()

dates = paired_df['Sample Date'].to_numpy().copy()

latitudes = paired_df['Latitude'].to_numpy().copy()
longitudes = paired_df['Longitude'].to_numpy().copy()

features_soto = paired_df[['chl_ocx', 'nflh', 'Rrs_443', 'Rrs_555']].to_numpy().copy()
features_aminRBD = paired_df[['Rrs_667', 'Rrs_678']].to_numpy().copy()
features_aminRBDKBBI = paired_df[['Rrs_667', 'Rrs_678']].to_numpy().copy()
features_stumpf = paired_df[['Sample Date', 'chlor_a']].to_numpy().copy()
features_shehhi = paired_df[['nflh']].to_numpy().copy()
features_Tomlinson = paired_df[['Rrs_443', 'Rrs_488', 'Rrs_531']].to_numpy().copy()
features_cannizzaro2008 = paired_df[['chl_ocx', 'Rrs_443', 'Rrs_555']].to_numpy().copy()
features_cannizzaro2009 = paired_df[['chl_ocx', 'Rrs_443', 'Rrs_555']].to_numpy().copy()
features_lou = paired_df[['Rrs_443', 'Rrs_488', 'Rrs_555']].to_numpy().copy()

features = paired_df[features_to_use[1:-6]]
features_used = features_to_use[3:-7]

features = np.array(features.values)

if(normalization == 0):
	features = features[:, 2:-1]
elif(normalization == 1):
	features = convertFeaturesByDepth(features[:, 2:], features_to_use[3:-7])
elif(normalization == 2):
	features = convertFeaturesByPosition(features[:, :-1], features_to_use[3:-7])



#Filter out samples between 0 and the detection_limit
if(use_hill_test == 1):
	valid_inds = (red_tide == 0) | (red_tide > detection_limit)
	valid_inds = np.where(valid_inds == True)[0]

	red_tide = red_tide[valid_inds]
	dates = dates[valid_inds]
	latitudes = latitudes[valid_inds]
	longitudes = longitudes[valid_inds]
	features = features[valid_inds, :]
	features_soto = features_soto[valid_inds, :]
	features_aminRBD = features_aminRBD[valid_inds, :]
	features_aminRBDKBBI = features_aminRBDKBBI[valid_inds, :]
	features_stumpf = features_stumpf[valid_inds, :]
	features_shehhi = features_shehhi[valid_inds, :]
	features_Tomlinson = features_Tomlinson[valid_inds, :]
	features_cannizzaro2008 = features_cannizzaro2008[valid_inds, :]
	features_cannizzaro2009 = features_cannizzaro2009[valid_inds, :]
	features_lou = features_lou[valid_inds, :]




if(use_nn_feature == 1):
	nn_classes = np.load('saved_model_info/'+configfilename+'/nn_classes.npy')
	features = np.concatenate((features, np.expand_dims(nn_classes, axis=1)), axis=1)
	features_used.append('nearest_ground_truth')
if(use_nn_feature == 2):
	knn_concs = np.load('saved_model_info/'+configfilename+'/knn_concs.npy')
	features = np.concatenate((features, np.expand_dims(knn_concs, axis=1)), axis=1)
	features_used.append('weighted_knn_conc')

concentrations = red_tide
classes = np.zeros((concentrations.shape[0], 1))

for i in range(len(classes)):
	if(concentrations[i] < detection_limit):
		classes[i] = 0
	else:
		classes[i] = 1

accs = np.zeros((len(randomseeds), 1))
confusonMatrixSum = np.zeros((num_classes, num_classes))
accsLinLee = np.zeros((len(randomseeds), 1))
permu_accs = np.zeros((len(randomseeds), len(features_used)))
accsNN = np.zeros((len(randomseeds), 1))

refFpr = []
tprs = []
modelThresholds = []
refFprNN = []
tprsNN = []
refFprKNN = []
tprsKNN = []
refFprHYCOM = []
tprsHYCOM = []
fprsSoto = []
tprsSoto = []
refFprAminRBD = []
tprsAminRBD = []
refFprStumpf = []
tprsStumpf = []
refFprShehhi = []
tprsShehhi = []
refFprTomlinson = []
tprsTomlinson = []
fprsCannizzaro2008 = []
tprsCannizzaro2008 = []
fprsCannizzaro2009 = []
tprsCannizzaro2009 = []
fprsAminRBDKBBI = []
tprsAminRBDKBBI = []
refFprLou = []
tprsLou = []

hycom_data = '/run/media/rfick/UF10/HYCOM/expt_50.1_netCDF/'

filename_lon = '/run/media/rfick/UF10/HYCOM/expt_50.1_lon.npy'
filename_lat = '/run/media/rfick/UF10/HYCOM/expt_50.1_lat.npy'

hycom_dates = np.load('/run/media/rfick/UF10/HYCOM/expt_50.1_dates.npy')

beta = 1

for model_number in range(len(randomseeds)):
	print('Model: {}'.format(model_number))

	reducedInds = np.load('saved_model_info/'+configfilename+'/reducedInds{}.npy'.format(model_number))

	usedClasses = classes[reducedInds]

	featuresTensor = torch.tensor(features)
	reducedFeaturesTensor = featuresTensor[reducedInds, :]

	reducedFeaturesSoto = features_soto[reducedInds, :]
	reducedFeaturesAminRBD = features_aminRBD[reducedInds, :]
	reducedFeaturesAminRBDKBBI = features_aminRBDKBBI[reducedInds, :]
	reducedFeaturesStumpf = features_stumpf[reducedInds, :]
	reducedFeaturesShehhi = features_shehhi[reducedInds, :]
	reducedFeaturesTomlinson = features_Tomlinson[reducedInds, :]
	reducedFeaturesCannizzaro2008 = features_cannizzaro2008[reducedInds, :]
	reducedFeaturesCannizzaro2009 = features_cannizzaro2009[reducedInds, :]
	reducedFeaturesLou = features_lou[reducedInds, :]

	reducedDates = dates[reducedInds]
	reducedLatitudes = latitudes[reducedInds]
	reducedLongitudes = longitudes[reducedInds]

	testInds = np.load('saved_model_info/'+configfilename+'/testInds{}.npy'.format(model_number))

	reducedDates = reducedDates[testInds]
	reducedLatitudes = reducedLatitudes[testInds]
	reducedLongitudes = reducedLongitudes[testInds]

	testSet = reducedFeaturesTensor[testInds, :].float().cuda()

	testSetSoto = reducedFeaturesSoto[testInds, :].astype(float)
	testSetAminRBD = reducedFeaturesAminRBD[testInds, :].astype(float)
	testSetAminRBDKBBI = reducedFeaturesAminRBDKBBI[testInds, :].astype(float)
	testSetStumpf = reducedFeaturesStumpf[testInds, :]
	testSetShehhi = reducedFeaturesShehhi[testInds, :].astype(float)
	testSetTomlinson = reducedFeaturesTomlinson[testInds, :].astype(float)
	testSetCannizzaro2008 = reducedFeaturesCannizzaro2008[testInds, :].astype(float)
	testSetCannizzaro2009 = reducedFeaturesCannizzaro2009[testInds, :].astype(float)
	testSetLou = reducedFeaturesLou[testInds, :].astype(float)

	outputSoto = SotoEtAlDetector(testSetSoto)
	outputAminRBD = AminEtAlRBDDetector(testSetAminRBD)
	outputAminRBDKBBI = AminEtAlRBDKBBIDetector(testSetAminRBDKBBI)
	outputStumpf = StumpfEtAlDetector(testSetStumpf)
	outputShehhi = ShehhiEtAlDetector(testSetShehhi)
	outputTomlinson = Tomlinson2009EtAlDetector(testSetTomlinson)
	outputCannizzaro2008 = Cannizzaro2008EtAlDetector(testSetCannizzaro2008)
	outputCannizzaro2009 = Cannizzaro2009EtAlDetector(testSetCannizzaro2009)
	outputLou = LouEtAlDetector(testSetLou)

	testClasses = usedClasses[testInds]

	testClasses = testClasses.astype(int)

	predictor = Predictor(testSet.shape[1], num_classes).cuda()
	predictor.load_state_dict(torch.load('saved_model_info/'+configfilename+'/predictor{}.pt'.format(model_number)))
	predictor.eval()

	output = predictor(testSet)

	output = output.detach().cpu().numpy()

	reducedDates = pd.DatetimeIndex(reducedDates)
	nn_preds = np.zeros((output.shape[0]))
	knn_preds = np.zeros((output.shape[0]))
	knn_concs = np.zeros((output.shape[0]))
	nn_concs = np.zeros((output.shape[0]))
	#hycom_concs = np.zeros((output.shape[0]))
	#Do some nearest neighbor thing with the last week's samples
	for i in range(len(reducedDates)):
		if(i%100 == 0):
			print('{}/{}'.format(i, len(reducedDates)))

		searchdate = reducedDates[i]
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
			closestClasses = df_classes[week_prior_inds]
			negativeInds = np.where(closestClasses==0)[0]
			positiveInds = np.where(closestClasses==1)[0]

			idx = find_nearest_latlon(df_lats[week_prior_inds], df_lons[week_prior_inds], reducedLatitudes[i], reducedLongitudes[i])

			closestConc = df_concs[week_prior_inds][idx]
			nn_concs[i] = closestConc
			knn_concs[i] = np.sum(NN_weights*df_concs_log[week_prior_inds])
			if(closestConc < detection_limit):
				nn_preds[i] = 0
			else:
				nn_preds[i] = 1
			if(np.sum(NN_weights[negativeInds]) > np.sum(NN_weights[positiveInds])):
				knn_preds[i] = 0
			else:
				knn_preds[i] = 1


			### Do HYCOM distance things
			#previous_dates = pd.DatetimeIndex(df_dates['Sample Date'][week_prior_inds])

			#earlieststartdatetime = previous_dates[0]
			#for j in range(len(previous_dates)):
			#	if(previous_dates[j] < earlieststartdatetime):
			#		earlieststartdatetime = previous_dates[j]

			#found_file_search_ind = file_search_netCDF(earlieststartdatetime, hycom_dates)
			#found_file_target_ind = file_search_netCDF(searchdate, hycom_dates)

			#simulation_file_list = []
			#for j in range(found_file_search_ind, found_file_target_ind+1):
			#	simulation_file_list.append(hycom_data+'netCDF4_file'+str(j)+'.nc')

			#If parcels fails for some reason, use euclidean distance
			#try:
			#	hycom_distances = hycom_dist_parcels(simulation_file_list, previous_dates, df_lats[week_prior_inds], df_lons[week_prior_inds], searchdate, reducedLatitudes[i], reducedLongitudes[i], filename_lon, filename_lat)
			#except Exception as e:
			#	print(e)
			#	hycom_distances = physicalDistance/100

			#Remove nans and infs if they somehow slip through
			#hycom_badidx = [k for k, arr in enumerate(hycom_distances) if not np.isfinite(arr).all()]
			#hycom_distances[hycom_badidx] = physicalDistance[hycom_badidx]/100

			#hycom_zeroidx = [k for k, arr in enumerate(hycom_distances) if hycom_distances[k]==0]
			#hycom_distances[hycom_zeroidx] = 0.5

			#inverseDistance = 1/hycom_distances
			#NN_weights = inverseDistance/np.sum(inverseDistance)
			#hycom_concs[i] = np.sum(NN_weights*df_concs_log[week_prior_inds])
		else:
			nn_concs[i] = 0
			nn_preds[i] = 0
			knn_preds[i] = 0

	accs[model_number] = accuracy_score(testClasses, np.argmax(output, axis=1))
	confusonMatrixSum += confusion_matrix(testClasses, np.argmax(output, axis=1))
	accsLinLee[model_number] = accuracy_score(testClasses, outputSoto)
	accsNN[model_number] = accuracy_score(testClasses, nn_preds)

	fprsSoto, tprsSoto = addNewModelResultsThreshold(outputSoto, testClasses, fprsSoto, tprsSoto)
	fprsCannizzaro2008, tprsCannizzaro2008 = addNewModelResultsThreshold(outputCannizzaro2008, testClasses, fprsCannizzaro2008, tprsCannizzaro2008)
	fprsCannizzaro2009, tprsCannizzaro2009 = addNewModelResultsThreshold(outputCannizzaro2009, testClasses, fprsCannizzaro2009, tprsCannizzaro2009)
	fprsAminRBDKBBI, tprsAminRBDKBBI = addNewModelResultsThreshold(outputAminRBDKBBI, testClasses, fprsAminRBDKBBI, tprsAminRBDKBBI)

	refFpr, tprs = addNewModelResultsROC(output[:, 1], testClasses, refFpr, tprs)
	refFprNN, tprsNN = addNewModelResultsROC(nn_concs, testClasses, refFprNN, tprsNN)
	refFprKNN, tprsKNN = addNewModelResultsROC(knn_concs, testClasses, refFprKNN, tprsKNN)
	#refFprHYCOM, tprsHYCOM = addNewModelResultsROC(hycom_concs, testClasses, refFprHYCOM, tprsHYCOM)
	refFprStumpf, tprsStumpf = addNewModelResultsROC(outputStumpf, testClasses, refFprStumpf, tprsStumpf)
	refFprShehhi, tprsShehhi = addNewModelResultsROC(outputShehhi, testClasses, refFprShehhi, tprsShehhi)
	refFprAminRBD, tprsAminRBD = addNewModelResultsROC(outputAminRBD, testClasses, refFprAminRBD, tprsAminRBD)
	refFprTomlinson, tprsTomlinson = addNewModelResultsROC(outputTomlinson, testClasses, refFprTomlinson, tprsTomlinson)
	refFprLou, tprsLou = addNewModelResultsROC(outputLou, testClasses, refFprLou, tprsLou)

	feature_permu = np.random.permutation(testSet.shape[0])
	for i in range(testSet.shape[1]):
		# permute feature i
		testSetClone = testSet.clone()
		testSetClone[:, i] = testSetClone[feature_permu, i]

		output = predictor(testSetClone)

		output = output.detach().cpu().numpy()

		output = np.argmax(output, axis=1)
		
		acc = accuracy_score(testClasses, output)

		permu_accs[model_number, i] = accs[model_number] - acc

feature_importance = np.zeros(testSet.shape[1])
for i in range(testSet.shape[1]):
	feature_importance[i] = np.mean(permu_accs[:, i])

inds = np.argsort(feature_importance)
inds = np.flip(inds)
for i in range(testSet.shape[1]):
	print('{}: {}'.format(features_used[inds[i]], feature_importance[inds[i]]))

filename_roc_curve_info = 'roc_curve_info'

refFpr = np.expand_dims(refFpr, axis=1)
fpr_and_tprs = np.concatenate((refFpr, tprs), axis=1)

np.save(filename_roc_curve_info+'/'+configfilename+'.npy', fpr_and_tprs)

refFprNN = np.expand_dims(refFprNN, axis=1)
fpr_and_tprsNN = np.concatenate((refFprNN, tprsNN), axis=1)

np.save(filename_roc_curve_info+'/'+configfilename+'_NN.npy', fpr_and_tprsNN)

refFprKNN = np.expand_dims(refFprKNN, axis=1)
fpr_and_tprsKNN = np.concatenate((refFprKNN, tprsKNN), axis=1)

np.save(filename_roc_curve_info+'/'+configfilename+'_KNN.npy', fpr_and_tprsKNN)

#refFprHYCOM = np.expand_dims(refFprHYCOM, axis=1)
#fpr_and_tprsHYCOM = np.concatenate((refFprHYCOM, tprsHYCOM), axis=1)

#np.save(filename_roc_curve_info+'/'+configfilename.split('_')[0]+'_HYCOM.npy', fpr_and_tprsHYCOM)

refFprStumpf = np.expand_dims(refFprStumpf, axis=1)
fpr_and_tprsStumpf = np.concatenate((refFprStumpf, tprsStumpf), axis=1)

np.save(filename_roc_curve_info+'/'+configfilename+'_Stumpf.npy', fpr_and_tprsStumpf)

refFprShehhi = np.expand_dims(refFprShehhi, axis=1)
fpr_and_tprsShehhi = np.concatenate((refFprShehhi, tprsShehhi), axis=1)

np.save(filename_roc_curve_info+'/'+configfilename+'_Shehhi.npy', fpr_and_tprsShehhi)

refFprAminRBD = np.expand_dims(refFprAminRBD, axis=1)
fpr_and_tprsAminRBD = np.concatenate((refFprAminRBD, tprsAminRBD), axis=1)

np.save(filename_roc_curve_info+'/'+configfilename+'_AminRBD.npy', fpr_and_tprsAminRBD)

refFprTomlinson = np.expand_dims(refFprTomlinson, axis=1)
fpr_and_tprsTomlinson = np.concatenate((refFprTomlinson, tprsTomlinson), axis=1)

np.save(filename_roc_curve_info+'/'+configfilename+'_Tomlinson.npy', fpr_and_tprsTomlinson)

refFprLou = np.expand_dims(refFprLou, axis=1)
fpr_and_tprsLou = np.concatenate((refFprLou, tprsLou), axis=1)

np.save(filename_roc_curve_info+'/'+configfilename+'_Lou.npy', fpr_and_tprsLou)

fpr_and_tprsSoto = np.zeros(2)
fpr_and_tprsSoto[0] = np.mean(fprsSoto)
fpr_and_tprsSoto[1] = np.mean(tprsSoto)

np.save(filename_roc_curve_info+'/'+configfilename+'_Soto.npy', fpr_and_tprsSoto)

fpr_and_tprsCannizzaro2008 = np.zeros(2)
fpr_and_tprsCannizzaro2008[0] = np.mean(fprsCannizzaro2008)
fpr_and_tprsCannizzaro2008[1] = np.mean(tprsCannizzaro2008)

np.save(filename_roc_curve_info+'/'+configfilename+'_Cannizzaro2008.npy', fpr_and_tprsCannizzaro2008)

fpr_and_tprsCannizzaro2009 = np.zeros(2)
fpr_and_tprsCannizzaro2009[0] = np.mean(fprsCannizzaro2009)
fpr_and_tprsCannizzaro2009[1] = np.mean(tprsCannizzaro2009)

np.save(filename_roc_curve_info+'/'+configfilename+'_Cannizzaro2009.npy', fpr_and_tprsCannizzaro2009)

fpr_and_tprsAminRBDKBBI = np.zeros(2)
fpr_and_tprsAminRBDKBBI[0] = np.mean(fprsAminRBDKBBI)
fpr_and_tprsAminRBDKBBI[1] = np.mean(tprsAminRBDKBBI)

np.save(filename_roc_curve_info+'/'+configfilename+'_AminRBDKBBI.npy', fpr_and_tprsAminRBDKBBI)