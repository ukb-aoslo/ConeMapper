% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


% Code to Find cone positions in a new set of images using a pretrained
% network and parameters

time_start = clock;
[ cnnFloderName, cnnCalcType, isCanceled ] = SelectVersionCNN();
if isCanceled
    return;
end

% Set-up MatConVNetPaths
BasePath = GetRootPath();
MatConvNetPath = fullfile(BasePath,cnnFloderName);
run(fullfile(MatConvNetPath,'matlab','vl_setupnn.m'))

% choose dataset with already trained cnn and detection parameters
% DataSet = 'confocal'; % original cases: 'confocal' or 'split detector'
% DataSet = 'g1+cunefare';
DataSet = 'alex training set';


% 14111 - Training (all 14 --> 224 grader 1) Validation (1 Image --> 16 grader 1)
% g1+cunefare - Training (all 14 --> 224 grader 1 + 100 selected from Cunefare Validation) Validation (1 Image --> 16 grader 1 + 16 selected from Cunefare Training)

% load in parameters
 params = get_parameters_Cone_CNN(DataSet);

% Choose Folder of images to detect cones in
ImageDir =  fullfile(BasePath, 'Images and Results\AlexTrainingSet\ToDetectCones');
% 'C:\Users\Jenny\Documents\MATLAB\CNN-Cone-Detection\Images and Results\Confocal\Test Images_150';

% format of images (must be readable by imread, must be 2D/grayscale format)
ImExtension = '.tif';

% Choose Folder to save coordinate
SaveDir = fullfile(BasePath, 'Images and Results\AlexTrainingSet\ToDetectCones');
% 'C:\Users\Jenny\Documents\MATLAB\CNN-Cone-Detection\Images and Results\Confocal\Test CNNcoord 42';



 
 SaveNewSetCones(params,ImageDir,ImExtension,SaveDir, cnnCalcType)
 
 time_start;
 time_finished = clock;