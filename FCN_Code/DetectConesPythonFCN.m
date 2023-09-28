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
    res = 1;

    % get absolute path to given path
    pathes = dir(path);
    if isempty(path)
        msgbox("Incorrect path", "Error", "error");
        return;
    end

    path = pathes(1).folder;

    % create waitbar
    wb = waitbar(0, 'FCN is running. Please, wait...');
    % get all children (including hidden)
    wbch = allchild(wb);
    % do Java magic
    jp = wbch(1).JavaPeer;
    jp.setIndeterminate(1);

    % create Python environment
    pyenv1 = pyenv("ExecutionMode","OutOfProcess");
    
    folderPath = [path, filesep(), 'tempData'];
    [~, ~, ~] = mkdir(folderPath);

    % get current path
    currentFile = mfilename( 'fullpath' );
    [currentPath, ~, ~] = fileparts( currentFile );
    % get path to python script
    pathToScript = [currentPath, filesep(), 'cone-detection-master-thesis'];
    cd(pathToScript);

    dividerH = 2;
    dividerW = 2;
    [height, width, ~] = size(image1);
    % if it is taking too much memory
    if CheckMemoryConsumingFCN(height, width)
        % divide image into pieces

        dev = gpuDevice();
        maxMemory = dev.AvailableMemory / 1024 - 1024 * 1024;

        coef = 1.435;
        freeAddition = 1.045e+06;

        pixels = (maxMemory - freeAddition) / coef;

        % find the height and width for a 1 piece
        actualPixels = height * width;
        newHeight = height;
        newWidth = width;
        while (actualPixels > pixels)
            newHeight = height / dividerH;
            actualPixels =  newHeight * newWidth;

            if (actualPixels < pixels)
                break;
            end

            newWidth = width / dividerW;
            actualPixels =  newHeight * newWidth;

            if (actualPixels < pixels)
                break;
            end
            
            dividerW = dividerW + 1;
            dividerH = dividerH + 1;
        end
    end

    dividerH = dividerH - 1;
    dividerW = dividerW - 1;
    tileHeight = floor(height / dividerH);
    tileWidth = floor(width / dividerW);

    currentTileStartH = 1;
    currentTileStartW = 1;

    currentTileEndH = 1;
    currentTileEndW = 1;

    % allocate the output variables
    data = zeros(0, 2);
    probMap = zeros(height, width);


    try
        % for each image tile
        for iTile = 1:dividerH*dividerW
            % find the row and column
            iRow = floor((iTile - 1) / dividerW) + 1;
            iColumn = iTile - (iRow - 1) * dividerW;
    
            % find the indexes to get the tile
            if (iColumn == 1)
                currentTileStartW = 1;
                if (iRow ~= 1)
                    currentTileStartH = currentTileStartH + tileHeight + 1;
                end
                currentTileEndH = currentTileStartH + tileHeight;
            else
                currentTileStartW = currentTileStartW + tileWidth + 1;
            end
            currentTileEndW = currentTileStartW + tileWidth;
    
            if (iRow == dividerH)
                currentTileEndH = height;
            end
            if (iColumn == dividerW)
                currentTileEndW = width;
            end
    
            % cut tile
            imageTile = image1( ...
                currentTileStartH:currentTileEndH, ...
                currentTileStartW:currentTileEndW);
    
            pathToImage = [path, filesep(), 'tempData', filesep(), 'imageTile_', num2str(iRow), '_', num2str(iColumn), '.tif'];
            imwrite(imageTile, pathToImage);
        
            % run script
            [output_path_image, output_path, res_pyhton] = pyrunfile(...
                ['detect.py ', ['"', pathToImage, '"'], ' -o ', ['"', folderPath, '"'], ' -matlab'],...
                ["output_path_image", "output_path", "result"]);
            
            % if result was successful
            if res_pyhton
                res = res && res_pyhton;
                dataPython = readmatrix(char(output_path));
                dataPython(:, 1) = dataPython(:, 1) + currentTileStartW - 1;
                dataPython(:, 1) = dataPython(:, 1) + currentTileStartH - 1;
                data = [data; dataPython];
                probMapPython = imread(char(output_path_image));
        
                [heightTile, widthTile, ~] = size(imageTile);
                [heightP, widthP, ~] = size(probMapPython);
                minHeight = min([heightTile, heightP]);
                minWidth = min([widthTile, widthP]);
                probMapGray = uint8(zeros(heightTile, widthTile));
                probMapGray(1:minHeight, 1:minWidth) = rgb2gray(probMapPython);
        
                probMapCurrent = probMapGray;
                imwrite(probMapCurrent, char(output_path_image));
                probMap( ...
                    currentTileStartH:currentTileEndH, ...
                    currentTileStartW:currentTileEndW) = probMapCurrent;
            end
        
        end
    catch ME
        disp(ME);
        res = 0;
    end

    if res
        pathToImage = [path, filesep(), 'tempData', filesep(), 'imageProbMap.tif'];
        imwrite(probMap, pathToImage);
    end
    % go back to current path
    cd(currentPath);
    terminate(pyenv1);
    close(wb);
end

function isExceeding = CheckMemoryConsumingFCN(height, width)
    isExceeding = false;

    dev = gpuDevice();
    % prediction in Kilobyte
    coef = 1.435;
    freeAddition = 1.045e+06;
    prediction = coef  * height * width + freeAddition;

    % if availiable memory is greater then prediction + 1GB
    if (dev.AvailableMemory / 1024 - prediction - 1024*1024) < 0
        % then memory will be exceeded
        isExceeding = true;
    end
end