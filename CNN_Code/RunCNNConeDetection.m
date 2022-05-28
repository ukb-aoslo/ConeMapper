% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


% code to run CNN based cone detection from beginning to end

% make sure matconvnet is installed and setup (see
% http://www.vlfeat.org/matconvnet/quick/)

% Make sure matconvnet is in the search path (can be added manually or by
% running vl_setupnn.m in MatConvNet\matconvnet-1.0-beta25cpu\matconvnet-1.0-beta25cpu\matlab)

clear
close all
time_start = clock;

[ cnnFloderName, cnnCalcType, isCanceled ] = SelectVersionCNN();
if isCanceled
    return;
end

% Set-up MatConVNetPaths
BasePath = GetRootPath();
MatConvNetPath = fullfile(BasePath, cnnFloderName);
run(fullfile(MatConvNetPath,'matlab','vl_setupnn.m'))

% Choose Data Set ('split detector' and 'confocal' use data sets in Cunefare 
% et. al 2017) to use a different data set add new class and modify called
% functions

% DataSet = 'split detector';
% DataSet = 'confocal'; % original cases: 'confocal' or 'split detector'
% DataSet = 'g1+cunefare';
% DataSet = 'alex training set';
% DataSet = '41 eye jlr set (black cut)';
DataSet = 'julius montages';

% load in parameters
 params = get_parameters_Cone_CNN(DataSet);


%% Create image patch database
CreateConeIMDB(params)


%% train network
% number of gpus to use
gpus = 1; 
 
cnn_Cones(gpus,params, cnnCalcType);

%% Save Probability Maps for training and validation data sets
TrainFlag = 1; % Save training data probability maps
ValidateFlag = 1; % Save Validation data probability maps (necesary for SaveValidationCones.m)

SaveProbabilityMaps(params,TrainFlag,ValidateFlag, cnnCalcType)

%% Find best combination of detection parameters
OptomizeConeDetectionParameters(params)

%% Find and save cones in the validation set

SaveValidationCones(params)

time_finished = clock;