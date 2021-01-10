NewDir = 'D:\PhD\CNN_NewTraining_Data\Training_Data_for_NEW_Training';
ImExtension = '.mat';
% load in list of images
ImageList = dir(fullfile(NewDir,['*' ImExtension])); 
ImageList =  {ImageList.name};

numFiles = length(ImageList);

SaveDir = fullfile('D:\PhD\CM', 'Images and Results\AlexTrainingSet');
% Make the folder for saving data
if(~exist(SaveDir,'dir'))
    mkdir(SaveDir);
end

% Loop through all images in training set
for iFile = 1:numFiles
    divide_data(fullfile(NewDir,ImageList{iFile}), SaveDir);
end