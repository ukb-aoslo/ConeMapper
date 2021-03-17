NewDir = 'D:\PhD\CNN_NewTraining_Data\Training_Data_for_NEW_Training';
ImExtension = '.mat';
% load in list of images
ImageList = dir(fullfile(NewDir,['*' ImExtension])); 
ImageList =  {ImageList.name};

numFiles = length(ImageList);

brighnesses = zeros(numFiles, 6);

% Loop through all images in training set
for iFile = 1:numFiles
    [minB, maxB, avgB, minBorderB, maxBorderB, avgBorderB] = get_brightness_stats(fullfile(NewDir,ImageList{iFile}));
    brighnesses(iFile, :) = [minB, maxB, avgB, minBorderB, maxBorderB, avgBorderB];
end

brighnesses(:, 7) = brighnesses(:, 1) > brighnesses(:, 4);
brighnesses(:, 8) = brighnesses(:, 2) > brighnesses(:, 5);
brighnesses(:, 9) = brighnesses(:, 3) > brighnesses(:, 6);

colNames = {'minCenter','maxCenter','avgCenter',  ...
    'minBorder','maxBorder','avgBorder', ... 
    'minCenGreaterMinBor','maxCenGreaterMaxBor','avgCenGreaterAvgBor'};
sTable = array2table(brighnesses,'VariableNames',colNames);

disp(sTable);

disp(min(brighnesses(:, 1)));
disp(max(brighnesses(:, 1)));
disp(min(brighnesses(:, 4)));
disp(max(brighnesses(:, 4)));