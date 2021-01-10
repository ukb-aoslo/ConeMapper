NewDir = 'D:\PhD\CNN_NewTraining_Data\Training_Data_2019_(g1+)cunefare';
ImExtension = '.tif';
CoordExtension = '.txt';
% load in list of images
ImageList = dir(fullfile(NewDir,['*' ImExtension])); 
ImageList = {ImageList.name};
numFiles = length(ImageList);

CoordList = dir(fullfile(NewDir,['*' CoordExtension])); 
CoordList = {CoordList.name};

SaveDir = fullfile('D:\PhD\CM', 'Images and Results\AlexTrainingSet');
ImageSaveDir =  fullfile(SaveDir, 'Training Images');
CoordSaveDir =  fullfile(SaveDir, 'Training Manual Coord');

% Loop through all images in training set
for iFile = 1:numFiles
    copyfile(fullfile(NewDir,ImageList{iFile}), ImageSaveDir)
    copyfile(fullfile(NewDir,CoordList{iFile}), CoordSaveDir)
end