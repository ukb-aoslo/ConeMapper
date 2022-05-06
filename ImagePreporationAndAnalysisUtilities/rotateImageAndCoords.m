folder = 'D:\MatLab\CM_Alex_git\ConeFinderApp\Images and Results\41eyeJLR\New folder\Training Images';
folderCoords = 'D:\MatLab\CM_Alex_git\ConeFinderApp\Images and Results\41eyeJLR\New folder\Training Manual Coord';
dirInfo = dir([folder, filesep(), '*.tif*']);
angles = [90, 180, 270];
anglesRod = [pi/2, pi, pi + pi/2];


for ind = 1:length(dirInfo)
    filename = [dirInfo(ind).folder, filesep(), dirInfo(ind).name];
    imageSource = imread(filename);
    csvFile = dir([folderCoords, filesep(), dirInfo(ind).name(1:end-5), '*.csv']);
    csvFilename = [csvFile.folder, filesep(), csvFile.name];
    coords = readmatrix(csvFilename);
    
    for indAng = 1:3
        imageRot = imrotate(imageSource, -angles(indAng));
        [x_rot, y_rot] = RotateCoords(coords(:, 1), coords(:, 2), size(imageRot), anglesRod(indAng), indAng);
        imwrite(imageRot, [filename(1:end-5), '_deg', num2str(angles(indAng)), '.tiff']);
        writematrix([x_rot, y_rot], [csvFilename(1:end-4), '_deg', num2str(angles(indAng)), '.csv']);
    end
end

function [x_rot, y_rot] = RotateCoords(x, y, fieldSize, theta, angleIndex)
% RotateCoords - rotates coordinates in field counterclockwise by theta angle
%   x - x coords of points in column
%   y - y coords of points in column
%   fieldSize - size of the field in format: fieldSize = [rows (height), cols (width)]
%   theta - angle in radians (pi/6 - 30 deg, pi/4 - 45 deg, pi/3 - 60 deg, pi/2 - 90 deg, pi - 180 deg)
    
    center = [round(fieldSize(2) / 2), round(fieldSize(1) / 2)];
    
    rotationMatrix = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    
    % do the rotation...
    v = [x,y];
    s = v - center;         % shift points in the plane so that the center of rotation is at the origin
    so = rotationMatrix*s';  % apply the rotation about the origin
    vo = so' + center;       % shift again so the origin goes back to the desired center of rotation
    
    x_rot = vo(:,1);
    y_rot = vo(:,2);
    
    if angleIndex == 1
        x_rot = x_rot + 1;
    elseif angleIndex == 2
        x_rot = x_rot + 1;
        y_rot = y_rot + 1;
    elseif angleIndex == 3
        y_rot = y_rot + 1;
    end
end