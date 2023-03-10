function [data, probMap, res] = DetectConesPythonFCN(image1, path)
%DETECTCONESPYTHONFCN Detects cones by FCN from python script
%   Input:
%   - image1 - grayscale image to detect cone.
%   - path - path for temp images.
%
%   Returs:
%   - data - detected cones in 2 columns array (x, y).
%   - probMap - grayscale image of probability map.
%   - res - 1 if detection was successfull, 0 otherwise.

    data = [];
    probMap = [];
    res = 0;

    % get absolute path to given path
    pathes = dir(path);
    if isempty(path)
        msgbox("Incorrect path", "Error", "error");
        return;
    end

    path = pathes(1).folder;

    % create waitbar
    wb = waitbar(0, 'Your Message Here');
    % get all children (including hidden)
    wbch = allchild(wb);
    % do Java magic
    jp = wbch(1).JavaPeer;
    jp.setIndeterminate(1);

    % create Python environment
    pyenv1 = pyenv("ExecutionMode","OutOfProcess");
    
    folderPath = [path, filesep(), 'tempData'];
    [~, ~, ~] = mkdir(folderPath);

    pathToImage = [path, filesep(), 'tempData', filesep(), 'image.tif'];
    imwrite(image1, pathToImage);

    % get current path
    currentFile = mfilename( 'fullpath' );
    [currentPath, ~, ~] = fileparts( currentFile );
    % get path to python script
    pathToScript = [currentPath, filesep(), 'cone-detection-master-thesis'];
    cd(pathToScript);

    % run script
    [ouptut, out2, res] = pyrunfile(...
        ['detect.py ', pathToImage, ' -t DT -o ', folderPath, ' -matlab'],...
        ["output_path_image", "output_path", "result"]);
    
    if res
        data = readmatrix(char(out2));
        probMap = imread(char(ouptut));

        [height, width, ~] = size(image1);
        [heightP, widthP, ~] = size(probMap);
        minHeight = min([height, heightP]);
        minWidth = min([width, widthP]);
        probMapGray = uint8(zeros(height, width));
        probMapGray(1:minHeight, 1:minWidth) = rgb2gray(probMap);

        probMap = probMapGray;
        imwrite(probMap, char(ouptut));
    end

    terminate(pyenv1);
    close(wb);
    % go back to current path
    cd(currentPath);
end

