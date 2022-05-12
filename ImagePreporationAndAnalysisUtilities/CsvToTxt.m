folderCoords = 'D:\MatLab\CM_Alex_git\ConeFinderApp\Images and Results\AlexTrainingSet\Validation Manual Coord';
dirInfo = dir([folderCoords, filesep(), '*.csv']);

for ind = 1:length(dirInfo)
    csvFilename = [dirInfo(ind).folder, filesep(), dirInfo(ind).name];
    coords = readmatrix(csvFilename);
    writematrix([coords(:, 1), coords(:, 2)], [csvFilename(1:end-4), '.txt']);
end
