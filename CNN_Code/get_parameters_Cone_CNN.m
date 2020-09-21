% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


function params = get_parameters_Cone_CNN(DataSet)
% function to return parameters for running cone CNN and paths to load and
% save data

%%%% General parameters
params.PatchSize = 33; % size of image patch


%%%% CNN training parameters
% Which network, 'lenet' default
params.CNNtrain.modeltype = 'lenet';

% Type of network, 'simplenn' (use for default) or 'dagnn' (not currently set up)
params.CNNtrain.networkType = 'simplenn' ;


%%%% Proabability map parameters
 % distance between patches extracted (currently only works for 1)
params.ProbMap.PatchDistance = 1;

% Number of batches to run at once, lower if running out of memory
params.ProbMap.batchsize = 2000;


%%%% Optimization parameters
params.opt.Sigma = [.1 .2 .3 .4 .5 .6 .7 .8 .9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2];
params.opt.PMthresh = [0 .1 .2 .3 .4 .5 .6 .7 .8 .9 1];
params.opt.ExtMaxH = [0 .05 .1 .15 .2 .25 .3 .4 .5];

% Percent of median manually marked distance between cones to search for
% matches
params.Opt.DistancePercent = .75;

% Number of pixels to remove from the sides of images when matching
params.Opt.BorderParams.HorizontalBorder = 7; 
params.Opt.BorderParams.VerticalBorder = 7;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Data Set Specific Parameters %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get Base FolderName
BasePath = GetRootPath();

% Set parameters based on data set
switch lower(DataSet)
    case 'split detector'
        
        %%%% General parameters
        % location of images and coordinate files
        params.ImageDirTrain = fullfile(BasePath,'Images and Results','Split Detector','Training Images');
        params.ManualCoordDirTrain = fullfile(BasePath,'Images and Results','Split Detector','Training Manual Coord');
        
        params.ImageDirValidate = fullfile(BasePath,'Images and Results','Split Detector','Validation Images');
        params.ManualCoordDirValidate = fullfile(BasePath,'Images and Results','Split Detector','Validation Manual Coord');
        
        % Extension on Images
        params.ImageExt ='.tif';
        
        % Text to add on to base of image names for coord files
        params.CoordAdditionalText = '_coords';
        
        % Format of coord file
        params.CoordExt = '.csv';
        
                
        %%%%% Parameters for imdb 
        % Set path to save imdb
        SaveName = 'imdb-SplitDetector-ConeCNN.mat';
        params.imdb.SavePath = fullfile(BasePath,'Images and Results','Split Detector',SaveName);
        
        
        %%%%% CNN training parameters
        % Set path to load imdb
        params.CNN.imdbPath =  params.imdb.SavePath;
        
        % Set path to save network training steps
        SaveNameNetTrain = 'CNN Training-Split Detector';
        params.CNN.TrainExpDir = fullfile(BasePath,'Images and Results','Split Detector',SaveNameNetTrain);
        
        % Set path to save final network
        SaveNameFinalNet = 'net-epoch-45-SplitDetector-ConeCNN.mat';
        params.CNN.NetworkSavePath = fullfile(BasePath,'Images and Results','Split Detector',SaveNameFinalNet);
        
        
        %%%% Proabability map parameters
        % Set path to load network
        params.ProbMap.NetworkPath =  params.CNN.NetworkSavePath;
        
        % Set paths to save probability maps
        params.ProbMap.SaveDirTrain  = fullfile(BasePath,'Images and Results','Split Detector','Probability Maps','Training');
        params.ProbMap.SaveDirValidate  = fullfile(BasePath,'Images and Results','Split Detector','Probability Maps','Validation');
        
        
        %%%% Optimization parameters
        % Set path to load training probability maps
        params.Opt.ProbMapDirTrain = params.ProbMap.SaveDirTrain;
        
        % Set path to save the optimization results
        SaveNameOpt = 'DetectionOptimization-SplitDetector-ConeCNN.mat';
        params.Opt.SavePath = fullfile(BasePath,'Images and Results','Split Detector',SaveNameOpt);
        
        
        %%%%% Validation result parameters
        % Set path for loading optimization results
        params.Results.OptimizationPath = params.Opt.SavePath;
        
        % Set path for saving detected cones from validation set
        SaveNameVal = 'Validation CNN Coord';
        params.Results.SaveDir = fullfile(BasePath,'Images and Results','Split Detector',SaveNameVal);
        
        % Set path to load validation probability maps
        params.Results.ProbMapDirValidate = params.ProbMap.SaveDirValidate;
        
        
    case 'confocal'
        
        %%%% General parameters
        % location of images and coordinate files
        params.ImageDirTrain = fullfile(BasePath,'Images and Results','Confocal','Training Images');
        params.ManualCoordDirTrain = fullfile(BasePath,'Images and Results','Confocal','Training Manual Coord');
        
        params.ImageDirValidate = fullfile(BasePath,'Images and Results','Confocal','Validation Images');
        params.ManualCoordDirValidate = fullfile(BasePath,'Images and Results','Confocal','Validation Manual Coord');

        % Extension on Images
        params.ImageExt ='.tif';
        
        % Text to add on to base of image names for coord files
        params.CoordAdditionalText = '_manualcoord';
        
        % Format of coord file
        params.CoordExt = '.txt';
        
        
        %%%%% Parameters for imdb 
        % Set path to save imdb
        SaveName = 'imdb-Confocal-ConeCNN.mat';
        params.imdb.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveName);
        
        
        %%%%% CNN training parameters
        % Set path to load imdb
        params.CNN.imdbPath =  params.imdb.SavePath;
        
        % Set path to save network training steps
        SaveNameNetTrain = 'CNN Training-Confocal';
        params.CNN.TrainExpDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameNetTrain);
        
        % Set path to save final network
        SaveNameFinalNet = 'net-epoch-45-Confocal-ConeCNN.mat';
        params.CNN.NetworkSavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameFinalNet);
        
        
        %%%% Proabability map parameters
        % Set path to load network
        params.ProbMap.NetworkPath =  params.CNN.NetworkSavePath;
        
        % Set paths to save probability maps
        params.ProbMap.SaveDirTrain  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Training');
        params.ProbMap.SaveDirValidate  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Validation');
        
        
        %%%% Optimization parameters
        % Set path to load training probability maps
        params.Opt.ProbMapDirTrain = params.ProbMap.SaveDirTrain;
        
        % Set path to save the optimization results
        SaveNameOpt = 'DetectionOptimization-Confocal-ConeCNN.mat';
        params.Opt.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameOpt);
        
        
        %%%%% Validation result parameters
        % Set path for loading optimization results
        params.Results.OptimizationPath = params.Opt.SavePath;
        
        % Set path for saving detected cones from validation set
        SaveNameVal = 'Validation CNN Coord';
        params.Results.SaveDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameVal);
        
        % Set path to load validation probability maps
        params.Results.ProbMapDirValidate = params.ProbMap.SaveDirValidate;
        
        % Set path to load scale_info for individual subjects   (JLR - 20.02.2019)
        SaveNameScale = 'scale_info_Confocal.csv';   % moved here from FindConfocalResults to allow for adjustment for diff data sets
        params.Results.Scale = fullfile(BasePath,'Images and Results','Confocal',SaveNameScale);
        
        
    case '2'        %'TrainCunefare_ValCunefare --> for reproducing results shown in the paper'
        
        %%%% General parameters
        % location of images and coordinate files
        params.ImageDirTrain = fullfile(BasePath,'Images and Results','Confocal','Training Images');
        params.ManualCoordDirTrain = fullfile(BasePath,'Images and Results','Confocal','Training Manual Coord');
        
        params.ImageDirValidate = fullfile(BasePath,'Images and Results','Confocal','Training Images');
        params.ManualCoordDirValidate = fullfile(BasePath,'Images and Results','Confocal','Training Manual Coord');

        % Extension on Images
        params.ImageExt = '.tif';
        
        % Text to add on to base of image names for coord files
        params.CoordAdditionalText = '';
        
        % Format of coord file
        params.CoordExt = '.txt';
        
        
        %%%%% Parameters for imdb 
        % Set path to save imdb
        SaveName = 'imdb-Confocal-ConeCNN.mat';
        params.imdb.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveName);
        
        
        %%%%% CNN training parameters
        % Set path to load imdb
        params.CNN.imdbPath =  params.imdb.SavePath;
        
        % Set path to save network training steps
        SaveNameNetTrain = 'CNN Training-Confocal';
        params.CNN.TrainExpDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameNetTrain);
        
        % Set path to save final network
        SaveNameFinalNet = 'net-epoch-45-Confocal-ConeCNN.mat';
        params.CNN.NetworkSavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameFinalNet);
        
        
        %%%% Proabability map parameters
        % Set path to load network
        params.ProbMap.NetworkPath =  params.CNN.NetworkSavePath;
        
        % Set paths to save probability maps
        params.ProbMap.SaveDirTrain  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Training');
        params.ProbMap.SaveDirValidate  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Validation');
        
        
        %%%% Optimization parameters
        % Set path to load training probability maps
        params.Opt.ProbMapDirTrain = params.ProbMap.SaveDirTrain;
        
        % Set path to save the optimization results
        SaveNameOpt = 'DetectionOptimization-Confocal-ConeCNN.mat';
        params.Opt.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameOpt);
        
        
        %%%%% Validation result parameters
        % Set path for loading optimization results
        params.Results.OptimizationPath = params.Opt.SavePath;
        
        % Set path for saving detected cones from validation set
        SaveNameVal = 'Training CNN Coord_checkNetwork';
        params.Results.SaveDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameVal);
        
        % Set path to load validation probability maps
        params.Results.ProbMapDirValidate = params.ProbMap.SaveDirValidate;
        
        % Set path to load scale_info for individual subjects   (JLR - 20.02.2019)
        SaveNameScale = 'scale_info_Confocal.csv';   % moved here from FindConfocalResults to allow for adjustment for diff data sets
        params.Results.Scale = fullfile(BasePath,'Images and Results','Confocal',SaveNameScale);
    
    case '3'        %'TrainCunefare_ValBonnAll'
        
        %%%% General parameters
        % location of images and coordinate files
        params.ImageDirTrain = fullfile(BasePath,'Images and Results','Confocal','Training Images');
        params.ManualCoordDirTrain = fullfile(BasePath,'Images and Results','Confocal','Training Manual Coord');
        
        params.ImageDirValidate = fullfile(BasePath,'Images and Results','Confocal','ValImages_BonnAll');
        params.ManualCoordDirValidate = fullfile(BasePath,'Images and Results','Confocal','ValManCoord_BonnAll');

        % Extension on Images
        params.ImageExt ='.tif';
        
        % Text to add on to base of image names for coord files
        params.CoordAdditionalText = '';
        
        % Format of coord file
        params.CoordExt = '.txt';
        
        
        %%%%% Parameters for imdb 
        % Set path to save imdb
        SaveName = 'imdb-Confocal-ConeCNN.mat';
        params.imdb.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveName);
        
        
        %%%%% CNN training parameters
        % Set path to load imdb
        params.CNN.imdbPath =  params.imdb.SavePath;
        
        % Set path to save network training steps
        SaveNameNetTrain = 'CNN Training-Confocal';
        params.CNN.TrainExpDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameNetTrain);
        
        % Set path to save final network
        SaveNameFinalNet = 'net-epoch-45-Confocal-ConeCNN.mat';
        params.CNN.NetworkSavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameFinalNet);
        
        
        %%%% Proabability map parameters
        % Set path to load network
        params.ProbMap.NetworkPath =  params.CNN.NetworkSavePath;
        
        % Set paths to save probability maps
        params.ProbMap.SaveDirTrain  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Training');
        params.ProbMap.SaveDirValidate  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Validation_BonnAll');
        
        
        %%%% Optimization parameters
        % Set path to load training probability maps
        params.Opt.ProbMapDirTrain = params.ProbMap.SaveDirTrain;
        
        % Set path to save the optimization results
        SaveNameOpt = 'DetectionOptimization-Confocal-ConeCNN.mat';
        params.Opt.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameOpt);
        
        
        %%%%% Validation result parameters
        % Set path for loading optimization results
        params.Results.OptimizationPath = params.Opt.SavePath;
        
        % Set path for saving detected cones from validation set
        SaveNameVal = 'ValCNNcoord_BonnAll';
        params.Results.SaveDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameVal);
        
        % Set path to load validation probability maps
        params.Results.ProbMapDirValidate = params.ProbMap.SaveDirValidate;
        
        % Set path to load scale_info for individual subjects   (JLR - 20.02.2019)
        SaveNameScale = 'scale_info_BonnAll.csv';   % moved here from FindConfocalResults to allow for adjustment for diff data sets
        params.Results.Scale = fullfile(BasePath,'Images and Results','Confocal',SaveNameScale);
        
    case '10411'        %'Train10-Grader1_Val4-Grader1'
        
        %%%% General parameters
        % location of images and coordinate files
        params.ImageDirTrain = fullfile(BasePath,'Images and Results','Confocal','Train10 Grader1 Images');
        params.ManualCoordDirTrain = fullfile(BasePath,'Images and Results','Confocal','Train10 Grader1 Manual Coord');
        
        params.ImageDirValidate = fullfile(BasePath,'Images and Results','Confocal','Val4 Grader1 Images');
        params.ManualCoordDirValidate = fullfile(BasePath,'Images and Results','Confocal','Val4 Grader1 Manual Coord');

        % Extension on Images
        params.ImageExt ='.tif';
        
        % Text to add on to base of image names for coord files
        params.CoordAdditionalText = '';
        
        % Format of coord file
        params.CoordExt = '.txt';
        
        
        %%%%% Parameters for imdb 
        % Set path to save imdb
        SaveName = 'imdb-Grader1-10train-ConeCNN.mat';
        params.imdb.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveName);
        
        
        %%%%% CNN training parameters
        % Set path to load imdb
        params.CNN.imdbPath =  params.imdb.SavePath;
        
        % Set path to save network training steps
        SaveNameNetTrain = 'CNN Training10-Grader1';
        params.CNN.TrainExpDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameNetTrain);
        
        % Set path to save final network
        SaveNameFinalNet = 'net-epoch-45-Grader1-10train-ConeCNN.mat';
        params.CNN.NetworkSavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameFinalNet);
        
        
        %%%% Proabability map parameters
        % Set path to load network
        params.ProbMap.NetworkPath =  params.CNN.NetworkSavePath;
        
        % Set paths to save probability maps
        params.ProbMap.SaveDirTrain  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Train10 Grader1');
        params.ProbMap.SaveDirValidate  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Val4 Grader1');
        
        
        %%%% Optimization parameters
        % Set path to load training probability maps
        params.Opt.ProbMapDirTrain = params.ProbMap.SaveDirTrain;
        
        % Set path to save the optimization results
        SaveNameOpt = 'DetectionOptimization-Grader1-10train-ConeCNN.mat';
        params.Opt.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameOpt);
        
        
        %%%%% Validation result parameters
        % Set path for loading optimization results
        params.Results.OptimizationPath = params.Opt.SavePath;
        
        % Set path for saving detected cones from validation set
        SaveNameVal = 'Val4 Grader1 CNN Coord';
        params.Results.SaveDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameVal);
        
        % Set path to load validation probability maps
        params.Results.ProbMapDirValidate = params.ProbMap.SaveDirValidate;
        
        % Set path to load scale_info for individual subjects   (JLR - 20.02.2019)
        SaveNameScale = 'scale_info_BonnAll.csv';   % moved here from FindConfocalResults to allow for adjustment for diff data sets
        params.Results.Scale = fullfile(BasePath,'Images and Results','Confocal',SaveNameScale);

    case '10412'        %'Train10-Grader1_Val4-Grader2'
        
        %%%% General parameters
        % location of images and coordinate files
        params.ImageDirTrain = fullfile(BasePath,'Images and Results','Confocal','Train10 Grader1 Images');
        params.ManualCoordDirTrain = fullfile(BasePath,'Images and Results','Confocal','Train10 Grader1 Manual Coord');
        
        params.ImageDirValidate = fullfile(BasePath,'Images and Results','Confocal','Val4 Grader2 Images');
        params.ManualCoordDirValidate = fullfile(BasePath,'Images and Results','Confocal','Val4 Grader2 Manual Coord');

        % Extension on Images
        params.ImageExt ='.tif';
        
        % Text to add on to base of image names for coord files
        params.CoordAdditionalText = '';
        
        % Format of coord file
        params.CoordExt = '.txt';
        
        
        %%%%% Parameters for imdb 
        % Set path to save imdb
        SaveName = 'imdb-Grader1-10train-ConeCNN.mat';
        params.imdb.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveName);
        
        
        %%%%% CNN training parameters
        % Set path to load imdb
        params.CNN.imdbPath =  params.imdb.SavePath;
        
        % Set path to save network training steps
        SaveNameNetTrain = 'CNN Training10-Grader1';
        params.CNN.TrainExpDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameNetTrain);
        
        % Set path to save final network
        SaveNameFinalNet = 'net-epoch-45-Grader1-10train-ConeCNN.mat';
        params.CNN.NetworkSavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameFinalNet);
        
        
        %%%% Proabability map parameters
        % Set path to load network
        params.ProbMap.NetworkPath =  params.CNN.NetworkSavePath;
        
        % Set paths to save probability maps
        params.ProbMap.SaveDirTrain  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Train10 Grader1');
        params.ProbMap.SaveDirValidate  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Val4 Grader2');
        
        
        %%%% Optimization parameters
        % Set path to load training probability maps
        params.Opt.ProbMapDirTrain = params.ProbMap.SaveDirTrain;
        
        % Set path to save the optimization results
        SaveNameOpt = 'DetectionOptimization-Grader1-10train-ConeCNN.mat';
        params.Opt.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameOpt);
        
        
        %%%%% Validation result parameters
        % Set path for loading optimization results
        params.Results.OptimizationPath = params.Opt.SavePath;
        
        % Set path for saving detected cones from validation set
        SaveNameVal = 'Val4 Grader2 CNN Coord';
        params.Results.SaveDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameVal);
        
        % Set path to load validation probability maps
        params.Results.ProbMapDirValidate = params.ProbMap.SaveDirValidate;
        
        % Set path to load scale_info for individual subjects   (JLR - 20.02.2019)
        SaveNameScale = 'scale_info_BonnAll.csv';   % moved here from FindConfocalResults to allow for adjustment for diff data sets
        params.Results.Scale = fullfile(BasePath,'Images and Results','Confocal',SaveNameScale);
        
    case '10422'        %'Train10-Grader2_Val4-Grader2'
        
        %%%% General parameters
        % location of images and coordinate files
        params.ImageDirTrain = fullfile(BasePath,'Images and Results','Confocal','Train10 Grader2 Images');
        params.ManualCoordDirTrain = fullfile(BasePath,'Images and Results','Confocal','Train10 Grader2 Manual Coord');
        
        params.ImageDirValidate = fullfile(BasePath,'Images and Results','Confocal','Val4 Grader2 Images');
        params.ManualCoordDirValidate = fullfile(BasePath,'Images and Results','Confocal','Val4 Grader2 Manual Coord');

        % Extension on Images
        params.ImageExt ='.tif';
        
        % Text to add on to base of image names for coord files
        params.CoordAdditionalText = '';
        
        % Format of coord file
        params.CoordExt = '.txt';
        
        
        %%%%% Parameters for imdb 
        % Set path to save imdb
        SaveName = 'imdb-Grader2-10train-ConeCNN.mat';
        params.imdb.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveName);
        
        
        %%%%% CNN training parameters
        % Set path to load imdb
        params.CNN.imdbPath =  params.imdb.SavePath;
        
        % Set path to save network training steps
        SaveNameNetTrain = 'CNN Training10-Grader2';
        params.CNN.TrainExpDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameNetTrain);
        
        % Set path to save final network
        SaveNameFinalNet = 'net-epoch-45-Grader2-10train-ConeCNN.mat';
        params.CNN.NetworkSavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameFinalNet);
        
        
        %%%% Proabability map parameters
        % Set path to load network
        params.ProbMap.NetworkPath =  params.CNN.NetworkSavePath;
        
        % Set paths to save probability maps
        params.ProbMap.SaveDirTrain  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Train10 Grader2');
        params.ProbMap.SaveDirValidate  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Val4 Grader2');
        
        
        %%%% Optimization parameters
        % Set path to load training probability maps
        params.Opt.ProbMapDirTrain = params.ProbMap.SaveDirTrain;
        
        % Set path to save the optimization results
        SaveNameOpt = 'DetectionOptimization-Grader2-10train-ConeCNN.mat';
        params.Opt.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameOpt);
        
        
        %%%%% Validation result parameters
        % Set path for loading optimization results
        params.Results.OptimizationPath = params.Opt.SavePath;
        
        % Set path for saving detected cones from validation set
        SaveNameVal = 'Val4 Grader2 CNN Coord';
        params.Results.SaveDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameVal);
        
        % Set path to load validation probability maps
        params.Results.ProbMapDirValidate = params.ProbMap.SaveDirValidate;
        
        % Set path to load scale_info for individual subjects   (JLR - 20.02.2019)
        SaveNameScale = 'scale_info_BonnAll.csv';   % moved here from FindConfocalResults to allow for adjustment for diff data sets
        params.Results.Scale = fullfile(BasePath,'Images and Results','Confocal',SaveNameScale);
        
    case '10421'        %'Train10-Grader2_Val4-Grader1'
        
        %%%% General parameters
        % location of images and coordinate files
        params.ImageDirTrain = fullfile(BasePath,'Images and Results','Confocal','Train10 Grader2 Images');
        params.ManualCoordDirTrain = fullfile(BasePath,'Images and Results','Confocal','Train10 Grader2 Manual Coord');
        
        params.ImageDirValidate = fullfile(BasePath,'Images and Results','Confocal','Val4 Grader1 Images');
        params.ManualCoordDirValidate = fullfile(BasePath,'Images and Results','Confocal','Val4 Grader1 Manual Coord');

        % Extension on Images
        params.ImageExt ='.tif';
        
        % Text to add on to base of image names for coord files
        params.CoordAdditionalText = '';
        
        % Format of coord file
        params.CoordExt = '.txt';
        
        
        %%%%% Parameters for imdb 
        % Set path to save imdb
        SaveName = 'imdb-Grader2-10train-ConeCNN.mat';
        params.imdb.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveName);
        
        
        %%%%% CNN training parameters
        % Set path to load imdb
        params.CNN.imdbPath =  params.imdb.SavePath;
        
        % Set path to save network training steps
        SaveNameNetTrain = 'CNN Training10-Grader2';
        params.CNN.TrainExpDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameNetTrain);
        
        % Set path to save final network
        SaveNameFinalNet = 'net-epoch-45-Grader2-10train-ConeCNN.mat';
        params.CNN.NetworkSavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameFinalNet);
        
        
        %%%% Proabability map parameters
        % Set path to load network
        params.ProbMap.NetworkPath =  params.CNN.NetworkSavePath;
        
        % Set paths to save probability maps
        params.ProbMap.SaveDirTrain  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Train10 Grader2');
        params.ProbMap.SaveDirValidate  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Val4 Grader1');
        
        
        %%%% Optimization parameters
        % Set path to load training probability maps
        params.Opt.ProbMapDirTrain = params.ProbMap.SaveDirTrain;
        
        % Set path to save the optimization results
        SaveNameOpt = 'DetectionOptimization-Grader2-10train-ConeCNN.mat';
        params.Opt.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameOpt);
        
        
        %%%%% Validation result parameters
        % Set path for loading optimization results
        params.Results.OptimizationPath = params.Opt.SavePath;
        
        % Set path for saving detected cones from validation set
        SaveNameVal = 'Val4 Grader1 CNN Coord';
        params.Results.SaveDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameVal);
        
        % Set path to load validation probability maps
        params.Results.ProbMapDirValidate = params.ProbMap.SaveDirValidate;
        
        % Set path to load scale_info for individual subjects   (JLR - 20.02.2019)
        SaveNameScale = 'scale_info_BonnAll.csv';   % moved here from FindConfocalResults to allow for adjustment for diff data sets
        params.Results.Scale = fullfile(BasePath,'Images and Results','Confocal',SaveNameScale);

    case '2041'        %'Train10+10-Grader1+2_Val4-Grader1'
        
        %%%% General parameters
        % location of images and coordinate files
        params.ImageDirTrain = fullfile(BasePath,'Images and Results','Confocal','Train10+10 Grader1+2 Images');
        params.ManualCoordDirTrain = fullfile(BasePath,'Images and Results','Confocal','Train10+10 Grader1+2 Manual Coord');
        
        params.ImageDirValidate = fullfile(BasePath,'Images and Results','Confocal','Val4 Grader1 Images');
        params.ManualCoordDirValidate = fullfile(BasePath,'Images and Results','Confocal','Val4 Grader1 Manual Coord');

        % Extension on Images
        params.ImageExt ='.tif';
        
        % Text to add on to base of image names for coord files
        params.CoordAdditionalText = '';
        
        % Format of coord file
        params.CoordExt = '.txt';
        
        
        %%%%% Parameters for imdb 
        % Set path to save imdb
        SaveName = 'imdb-Grader1+2-10+10train-ConeCNN.mat';
        params.imdb.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveName);
        
        
        %%%%% CNN training parameters
        % Set path to load imdb
        params.CNN.imdbPath =  params.imdb.SavePath;
        
        % Set path to save network training steps
        SaveNameNetTrain = 'CNN Training10+10-Grader1+2';
        params.CNN.TrainExpDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameNetTrain);
        
        % Set path to save final network
        SaveNameFinalNet = 'net-epoch-45-Grader1+2-10+10train-ConeCNN.mat';
        params.CNN.NetworkSavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameFinalNet);
        
        
        %%%% Proabability map parameters
        % Set path to load network
        params.ProbMap.NetworkPath =  params.CNN.NetworkSavePath;
        
        % Set paths to save probability maps
        params.ProbMap.SaveDirTrain  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Train10+10 Grader1+2');
        params.ProbMap.SaveDirValidate  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Val4 Grader1+2');
        
        
        %%%% Optimization parameters
        % Set path to load training probability maps
        params.Opt.ProbMapDirTrain = params.ProbMap.SaveDirTrain;
        
        % Set path to save the optimization results
        SaveNameOpt = 'DetectionOptimization-Grader1+2-10+10train-ConeCNN.mat';
        params.Opt.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameOpt);
        
        
        %%%%% Validation result parameters
        % Set path for loading optimization results
        params.Results.OptimizationPath = params.Opt.SavePath;
        
        % Set path for saving detected cones from validation set
        SaveNameVal = 'Val4 Grader1+2 CNN Coord';
        params.Results.SaveDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameVal);
        
        % Set path to load validation probability maps
        params.Results.ProbMapDirValidate = params.ProbMap.SaveDirValidate;
        
        % Set path to load scale_info for individual subjects   (JLR - 20.02.2019)
        SaveNameScale = 'scale_info_BonnAll.csv';   % moved here from FindConfocalResults to allow for adjustment for diff data sets
        params.Results.Scale = fullfile(BasePath,'Images and Results','Confocal',SaveNameScale);

    case '2042'        %'Train10+10-Grader1+2_Val4-Grader2'
        
        %%%% General parameters
        % location of images and coordinate files
        params.ImageDirTrain = fullfile(BasePath,'Images and Results','Confocal','Train10+10 Grader1+2 Images');
        params.ManualCoordDirTrain = fullfile(BasePath,'Images and Results','Confocal','Train10+10 Grader1+2 Manual Coord');
        
        params.ImageDirValidate = fullfile(BasePath,'Images and Results','Confocal','Val4 Grader2 Images');
        params.ManualCoordDirValidate = fullfile(BasePath,'Images and Results','Confocal','Val4 Grader2 Manual Coord');

        % Extension on Images
        params.ImageExt ='.tif';
        
        % Text to add on to base of image names for coord files
        params.CoordAdditionalText = '';
        
        % Format of coord file
        params.CoordExt = '.txt';
        
        
        %%%%% Parameters for imdb 
        % Set path to save imdb
        SaveName = 'imdb-Grader1+2-10+10train-ConeCNN.mat';
        params.imdb.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveName);
        
        
        %%%%% CNN training parameters
        % Set path to load imdb
        params.CNN.imdbPath =  params.imdb.SavePath;
        
        % Set path to save network training steps
        SaveNameNetTrain = 'CNN Training10+10-Grader1+2';
        params.CNN.TrainExpDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameNetTrain);
        
        % Set path to save final network
        SaveNameFinalNet = 'net-epoch-45-Grader1+2-10+10train-ConeCNN.mat';
        params.CNN.NetworkSavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameFinalNet);
        
        
        %%%% Proabability map parameters
        % Set path to load network
        params.ProbMap.NetworkPath =  params.CNN.NetworkSavePath;
        
        % Set paths to save probability maps
        params.ProbMap.SaveDirTrain  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Train10+10 Grader1+2');
        params.ProbMap.SaveDirValidate  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Val4 Grader1+2');
        
        
        %%%% Optimization parameters
        % Set path to load training probability maps
        params.Opt.ProbMapDirTrain = params.ProbMap.SaveDirTrain;
        
        % Set path to save the optimization results
        SaveNameOpt = 'DetectionOptimization-Grader1+2-10+10train-ConeCNN.mat';
        params.Opt.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameOpt);
        
        
        %%%%% Validation result parameters
        % Set path for loading optimization results
        params.Results.OptimizationPath = params.Opt.SavePath;
        
        % Set path for saving detected cones from validation set
        SaveNameVal = 'Val4 Grader2+1 CNN Coord';
        params.Results.SaveDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameVal);
        
        % Set path to load validation probability maps
        params.Results.ProbMapDirValidate = params.ProbMap.SaveDirValidate;
        
        % Set path to load scale_info for individual subjects   (JLR - 20.02.2019)
        SaveNameScale = 'scale_info_BonnAll.csv';   % moved here from FindConfocalResults to allow for adjustment for diff data sets
        params.Results.Scale = fullfile(BasePath,'Images and Results','Confocal',SaveNameScale);

    case '12'        %'compare grader1 (Manual) and grader2's (CNN) conelocs'
        
        %%%% General parameters
        % location of images and coordinate files
        params.ImageDirTrain = fullfile(BasePath,'Images and Results','Confocal','not needed');
        params.ManualCoordDirTrain = fullfile(BasePath,'Images and Results','Confocal','not needed');
        
        params.ImageDirValidate = fullfile(BasePath,'Images and Results','Confocal','not needed');
        params.ManualCoordDirValidate = fullfile(BasePath,'Images and Results','Confocal','Locs_AllGrader1_txtWMH');

        % Extension on Images
        params.ImageExt ='.tif';
        
        % Text to add on to base of image names for coord files
        params.CoordAdditionalText = '';
        
        % Format of coord file
        params.CoordExt = '.txt';
        
        
        %%%%% Parameters for imdb 
        % Set path to save imdb
        SaveName = 'imdb-Grader1-10train-ConeCNN.mat';
        params.imdb.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveName);
        
        
        %%%%% CNN training parameters
        % Set path to load imdb
        params.CNN.imdbPath = params.imdb.SavePath;
        
        % Set path to save network training steps
        SaveNameNetTrain = 'CNN Training10-Grader1';
        params.CNN.TrainExpDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameNetTrain);
        
        % Set path to save final network
        SaveNameFinalNet = 'net-epoch-45-Grader1-10train-ConeCNN.mat';
        params.CNN.NetworkSavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameFinalNet);
        
        
        %%%% Proabability map parameters
        % Set path to load network
        params.ProbMap.NetworkPath = params.CNN.NetworkSavePath;
        
        % Set paths to save probability maps
        params.ProbMap.SaveDirTrain = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Train10 Grader1');
        params.ProbMap.SaveDirValidate = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Val4 Grader1');
        
        
        %%%% Optimization parameters
        % Set path to load training probability maps
        params.Opt.ProbMapDirTrain = params.ProbMap.SaveDirTrain;
        
        % Set path to save the optimization results
        SaveNameOpt = 'DetectionOptimization-Grader1-10train-ConeCNN.mat';
        params.Opt.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameOpt);
        
        
        %%%%% Validation result parameters
        % Set path for loading optimization results
        params.Results.OptimizationPath = params.Opt.SavePath;
        
        % Set path for saving detected cones from validation set
        SaveNameVal = 'Locs_AllGrader2_mat';
        params.Results.SaveDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameVal);
        
        % Set path to load validation probability maps
        params.Results.ProbMapDirValidate = params.ProbMap.SaveDirValidate;
        
        % Set path to load scale_info for individual subjects   (JLR - 20.02.2019)
        SaveNameScale = 'scale_info_BonnAll.csv';   % moved here from FindConfocalResults to allow for adjustment for diff data sets
        params.Results.Scale = fullfile(BasePath,'Images and Results','Confocal',SaveNameScale);
    
    case '21'        %'compare grader2 (Manual) and grader1's (CNN) conelocs'
        
        %%%% General parameters
        % location of images and coordinate files
        params.ImageDirTrain = fullfile(BasePath,'Images and Results','Confocal','not needed');
        params.ManualCoordDirTrain = fullfile(BasePath,'Images and Results','Confocal','not needed');
        
        params.ImageDirValidate = fullfile(BasePath,'Images and Results','Confocal','not needed');
        params.ManualCoordDirValidate = fullfile(BasePath,'Images and Results','Confocal','Locs_AllGrader2_txtJLR');

        % Extension on Images
        params.ImageExt ='.tif';
        
        % Text to add on to base of image names for coord files
        params.CoordAdditionalText = '';
        
        % Format of coord file
        params.CoordExt = '.txt';
        
        
        %%%%% Parameters for imdb 
        % Set path to save imdb
        SaveName = 'imdb-Grader1-10train-ConeCNN.mat';
        params.imdb.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveName);
        
        
        %%%%% CNN training parameters
        % Set path to load imdb
        params.CNN.imdbPath = params.imdb.SavePath;
        
        % Set path to save network training steps
        SaveNameNetTrain = 'CNN Training10-Grader1';
        params.CNN.TrainExpDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameNetTrain);
        
        % Set path to save final network
        SaveNameFinalNet = 'net-epoch-45-Grader1-10train-ConeCNN.mat';
        params.CNN.NetworkSavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameFinalNet);
        
        
        %%%% Proabability map parameters
        % Set path to load network
        params.ProbMap.NetworkPath = params.CNN.NetworkSavePath;
        
        % Set paths to save probability maps
        params.ProbMap.SaveDirTrain = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Train10 Grader1');
        params.ProbMap.SaveDirValidate = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Val4 Grader1');
        
        
        %%%% Optimization parameters
        % Set path to load training probability maps
        params.Opt.ProbMapDirTrain = params.ProbMap.SaveDirTrain;
        
        % Set path to save the optimization results
        SaveNameOpt = 'DetectionOptimization-Grader1-10train-ConeCNN.mat';
        params.Opt.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameOpt);
        
        
        %%%%% Validation result parameters
        % Set path for loading optimization results
        params.Results.OptimizationPath = params.Opt.SavePath;
        
        % Set path for saving detected cones from validation set
        SaveNameVal = 'Locs_AllGrader1_mat';
        params.Results.SaveDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameVal);
        
        % Set path to load validation probability maps
        params.Results.ProbMapDirValidate = params.ProbMap.SaveDirValidate;
        
        % Set path to load scale_info for individual subjects   (JLR - 20.02.2019)
        SaveNameScale = 'scale_info_BonnAll.csv';   % moved here from FindConfocalResults to allow for adjustment for diff data sets
        params.Results.Scale = fullfile(BasePath,'Images and Results','Confocal',SaveNameScale);
        
    case '224x2'        % (JLR - 14.03.2019)
                        % 'Train with all Bonn marked mosaics we currently have,...
                        %  --> 14 eyes (2 x 224 150pixel images) grader 1 + grader 2,...
                        % should be updated when new mosaics are graded
        
        %%%% General parameters
        % location of images and coordinate files
        params.ImageDirTrain = fullfile(BasePath,'Images and Results','Confocal','224x2 - ALL Bonn Images150');
        params.ManualCoordDirTrain = fullfile(BasePath,'Images and Results','Confocal','224x2 - ALL Bonn ManualCoord150');
        
        params.ImageDirValidate = fullfile(BasePath,'Images and Results','Confocal','Validation Images');                 % Val with Cunefare data
        params.ManualCoordDirValidate = fullfile(BasePath,'Images and Results','Confocal','Validation Manual Coord');     % Val with Cunefare data

        % Extension on Images
        params.ImageExt ='.tif'; 
        
        % Text to add on to base of image names for coord files
        params.CoordAdditionalText = '';
        
        % Format of coord file
        params.CoordExt = '.txt';
        
        
        %%%%% Parameters for imdb 
        % Set path to save imdb
        SaveName = 'imdb-224x2-AllBonnTraining-ConeCNN.mat';
        params.imdb.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveName);
        
        
        %%%%% CNN training parameters
        % Set path to load imdb
        params.CNN.imdbPath =  params.imdb.SavePath;
        
        % Set path to save network training steps
        SaveNameNetTrain = 'CNN Training_224x2_AllBonn';
        params.CNN.TrainExpDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameNetTrain);
        
        % Set path to save final network
        SaveNameFinalNet = 'net-epoch-45-224x2AllBonnTraining-ConeCNN.mat';
        params.CNN.NetworkSavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameFinalNet);
        
        
        %%%% Proabability map parameters
        % Set path to load network
        params.ProbMap.NetworkPath =  params.CNN.NetworkSavePath;
        
        % Set paths to save probability maps
        params.ProbMap.SaveDirTrain  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Train 224x2 AllBonn');
        params.ProbMap.SaveDirValidate  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Val 224x2 AllBonn');
        
        
        %%%% Optimization parameters
        % Set path to load training probability maps
        params.Opt.ProbMapDirTrain = params.ProbMap.SaveDirTrain;
        
        % Set path to save the optimization results
        SaveNameOpt = 'DetectionOptimization-224x2_AllBonnTraining-ConeCNN.mat';
        params.Opt.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameOpt);
        
        
        %%%%% Validation result parameters
        % Set path for loading optimization results
        params.Results.OptimizationPath = params.Opt.SavePath;
        
        % Set path for saving detected cones from validation set
        SaveNameVal = 'Validation CNN Coord';
        params.Results.SaveDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameVal);
        
        % Set path to load validation probability maps
        params.Results.ProbMapDirValidate = params.ProbMap.SaveDirValidate;
        
        % Set path to load scale_info for individual subjects   
        SaveNameScale = 'scale_info_BonnAll.csv';   % moved here from FindConfocalResults to allow for adjustment for diff data sets
        params.Results.Scale = fullfile(BasePath,'Images and Results','Confocal',SaveNameScale);

    
    case '42'        % (JLR - 13.03.2019)
                     % 'Train with all marked mosaics we currently have --> 14 eyes (224 150pixel images) grader 1 + grader 2,...
                     % + Cunefare, Farsiu... 840 150pix mosaics'
                     % should be updated when new mosaics are graded
        
        %%%% General parameters
        % location of images and coordinate files
        params.ImageDirTrain = fullfile(BasePath,'Images and Results','Confocal','42-ALL Training Images150');
        params.ManualCoordDirTrain = fullfile(BasePath,'Images and Results','Confocal','42-ALL Training ManualCoord150');
        
        params.ImageDirValidate = fullfile(BasePath,'Images and Results','Confocal','Val1 Grader1 Images');                 % not valid in this case
        params.ManualCoordDirValidate = fullfile(BasePath,'Images and Results','Confocal','Val1 Grader1 Manual Coord');     % not valid in this case

        % Extension on Images
        params.ImageExt ='.tif';
        
        % Text to add on to base of image names for coord files
        params.CoordAdditionalText = '';
        
        % Format of coord file
        params.CoordExt = '.txt';
        
        
        %%%%% Parameters for imdb 
        % Set path to save imdb
        SaveName = 'imdb-42-ALLdataTraining-ConeCNN.mat';
        params.imdb.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveName);
        
        
        %%%%% CNN training parameters
        % Set path to load imdb
        params.CNN.imdbPath =  params.imdb.SavePath;
        
        % Set path to save network training steps
        SaveNameNetTrain = 'CNN Training_42_ALLdata';
        params.CNN.TrainExpDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameNetTrain);
        
        % Set path to save final network
        SaveNameFinalNet = 'net-epoch-45-42ALLdataTraining-ConeCNN.mat';
        params.CNN.NetworkSavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameFinalNet);
        
        
        %%%% Proabability map parameters
        % Set path to load network
        params.ProbMap.NetworkPath =  params.CNN.NetworkSavePath;
        
        % Set paths to save probability maps
        params.ProbMap.SaveDirTrain  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Train 42 ALLdata');
        params.ProbMap.SaveDirValidate  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Val 42 AlldataTraining');
        
        
        %%%% Optimization parameters
        % Set path to load training probability maps
        params.Opt.ProbMapDirTrain = params.ProbMap.SaveDirTrain;
        
        % Set path to save the optimization results
        SaveNameOpt = 'DetectionOptimization-42_ALLdataTraining-ConeCNN.mat';
        params.Opt.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameOpt);
        
        
        %%%%% Validation result parameters
        % Set path for loading optimization results
        params.Results.OptimizationPath = params.Opt.SavePath;
        
        % Set path for saving detected cones from validation set
        SaveNameVal = 'Val1 Grader1 CNN Coord';
        params.Results.SaveDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameVal);
        
        % Set path to load validation probability maps
        params.Results.ProbMapDirValidate = params.ProbMap.SaveDirValidate;
        
        % Set path to load scale_info for individual subjects   (JLR - 20.02.2019)
        SaveNameScale = 'scale_info_AllConfocal.csv';   % moved here from FindConfocalResults to allow for adjustment for diff data sets
        params.Results.Scale = fullfile(BasePath,'Images and Results','Confocal',SaveNameScale);
        
    case '14111'     % (JLR - 02.04.2019)
                     % 'Train with all grader 1 marked mosaics we currently have --> 14 eyes (224 150pixel images),...
                     % + validate with 1 extra BAK8044 mosaic
        
        %%%% General parameters
        % location of images and coordinate files
        params.ImageDirTrain = fullfile(BasePath,'Images and Results','Confocal','Train14 Grader1 Images');
        params.ManualCoordDirTrain = fullfile(BasePath,'Images and Results','Confocal','Train14 Grader1 Manual Coord');
        
        params.ImageDirValidate = fullfile(BasePath,'Images and Results','Confocal','Val1 Grader1 Images');                 % not valid in this case
        params.ManualCoordDirValidate = fullfile(BasePath,'Images and Results','Confocal','Val1 Grader1 Manual Coord');     % not valid in this case

        % Extension on Images
        params.ImageExt ='.tif';
        
        % Text to add on to base of image names for coord files
        params.CoordAdditionalText = '';
        
        % Format of coord file
        params.CoordExt = '.txt';
        
        
        %%%%% Parameters for imdb 
        % Set path to save imdb
        SaveName = 'imdb-14111-Training-ConeCNN.mat';
        params.imdb.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveName);
        
        
        %%%%% CNN training parameters
        % Set path to load imdb
        params.CNN.imdbPath =  params.imdb.SavePath;
        
        % Set path to save network training steps
        SaveNameNetTrain = 'CNN Training_14111';
        params.CNN.TrainExpDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameNetTrain);
        
        % Set path to save final network
        SaveNameFinalNet = 'net-epoch-45-14111Training-ConeCNN.mat';
        params.CNN.NetworkSavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameFinalNet);
        
        
        %%%% Proabability map parameters
        % Set path to load network
        params.ProbMap.NetworkPath =  params.CNN.NetworkSavePath;
        
        % Set paths to save probability maps
        params.ProbMap.SaveDirTrain  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Train 14111');
        params.ProbMap.SaveDirValidate  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Val 14111');
        
        
        %%%% Optimization parameters
        % Set path to load training probability maps
        params.Opt.ProbMapDirTrain = params.ProbMap.SaveDirTrain;
        
        % Set path to save the optimization results
        SaveNameOpt = 'DetectionOptimization-14111_Training-ConeCNN.mat';
        params.Opt.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameOpt);
        
        
        %%%%% Validation result parameters
        % Set path for loading optimization results
        params.Results.OptimizationPath = params.Opt.SavePath;
        
        % Set path for saving detected cones from validation set
        SaveNameVal = 'Val1 Grader1 CNN Coord';
        params.Results.SaveDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameVal);
        
        % Set path to load validation probability maps
        params.Results.ProbMapDirValidate = params.ProbMap.SaveDirValidate;
        
        % Set path to load scale_info for individual subjects   (JLR - 20.02.2019)
        SaveNameScale = 'scale_info_AllConfocal.csv';   % moved here from FindConfocalResults to allow for adjustment for diff data sets
        params.Results.Scale = fullfile(BasePath,'Images and Results','Confocal',SaveNameScale);        


    case 'g1+cunefare'      % (JLR - 10.04.2019)
                            % 'Train with all grader 1 marked mosaics we currently have --> 14 eyes (224 150pixel images)
                            % + 100 selected from Cunefare Validation Set,...
                            % + validate with 1 extra BAK8044 mosaic (16 150x150) + 16 Cunefare
        
        %%%% General parameters
        % location of images and coordinate files
        params.ImageDirTrain = fullfile(BasePath,'Images and Results','Confocal','Train g1+cunefare Images');
        params.ManualCoordDirTrain = fullfile(BasePath,'Images and Results','Confocal','Train g1+cunefare Manual Coord');
        
        params.ImageDirValidate = fullfile(BasePath,'Images and Results','Confocal','Val1 Grader1 Images');                 % not valid in this case
        params.ManualCoordDirValidate = fullfile(BasePath,'Images and Results','Confocal','Val1 Grader1 Manual Coord');     % not valid in this case

        % Extension on Images
        params.ImageExt ='.tif';
        
        % Text to add on to base of image names for coord files
        params.CoordAdditionalText = '';
        
        % Format of coord file
        params.CoordExt = '.txt';
        
        
        %%%%% Parameters for imdb 
        % Set path to save imdb
        SaveName = 'imdb-g1+cunefare-Training-ConeCNN.mat';
        params.imdb.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveName);
        
        
        %%%%% CNN training parameters
        % Set path to load imdb
        params.CNN.imdbPath =  params.imdb.SavePath;
        
        % Set path to save network training steps
        SaveNameNetTrain = 'CNN Training_g1+cunefare';
        params.CNN.TrainExpDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameNetTrain);
        
        % Set path to save final network
        SaveNameFinalNet = 'net-epoch-45-g1+cunefareTraining-ConeCNN.mat';
        params.CNN.NetworkSavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameFinalNet);
        
        
        %%%% Proabability map parameters
        % Set path to load network
        params.ProbMap.NetworkPath =  params.CNN.NetworkSavePath;
        
        % Set paths to save probability maps
        params.ProbMap.SaveDirTrain  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Train g1+cunefare');
        params.ProbMap.SaveDirValidate  = fullfile(BasePath,'Images and Results','Confocal','Probability Maps','Val g1+cunefare');
        
        
        %%%% Optimization parameters
        % Set path to load training probability maps
        params.Opt.ProbMapDirTrain = params.ProbMap.SaveDirTrain;
        
        % Set path to save the optimization results
        SaveNameOpt = 'DetectionOptimization-g1+cunefare_Training-ConeCNN.mat';
        params.Opt.SavePath = fullfile(BasePath,'Images and Results','Confocal',SaveNameOpt);
        
        
        %%%%% Validation result parameters
        % Set path for loading optimization results
        params.Results.OptimizationPath = params.Opt.SavePath;
        
        % Set path for saving detected cones from validation set
        SaveNameVal = 'Val1 Grader1 CNN Coord';
        params.Results.SaveDir = fullfile(BasePath,'Images and Results','Confocal',SaveNameVal);
        
        % Set path to load validation probability maps
        params.Results.ProbMapDirValidate = params.ProbMap.SaveDirValidate;
        
        % Set path to load scale_info for individual subjects   (JLR - 20.02.2019)
        SaveNameScale = 'scale_info_AllConfocal.csv';   % moved here from FindConfocalResults to allow for adjustment for diff data sets
        params.Results.Scale = fullfile(BasePath,'Images and Results','Confocal',SaveNameScale);        
        
        
    otherwise
        error('Please select a known data set or add your own case')      
end


end