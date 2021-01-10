function divide_data(filepath, SaveDir)
    % Fname = 'BAK1012R_2019_02_15_13_28_33_AOSLO_V004_stabilized_840_annotated_JLR';
    
    [folder, baseFileNameNoExt, ~] = fileparts(filepath);
    Fname = [folder '\' baseFileNameNoExt];
    mat_ext = '.mat';
    tiff_ext = '.tiff';
    I = 0;
    conelocs = 0;
    
    % load data
    load([Fname mat_ext]);
%     conecount = size(conelocs(conelocs(:,3)~=0,1),1);
    conelocs = unique(conelocs, 'rows', 'stable');
    boxposition;
    % TotalConeCount = size(conelocs,1);

    if multiple_mosaics == 1
        try
            ChangeMosaic = I;
            I = ChangeMosaic{1};
        catch
            ChangeMosaic = squeeze(num2cell(I,[1 2]));
            I = ChangeMosaic{1};
        end
    end
% 
%     imshow(I), title([Fname,'  ',num2str(conecount),' cones'],'Interpreter','none'), hold on
%     ax = gca;                    % get the current axis
%     Ttl = ax.Title;              % get the title text object
%     Ttl.FontSize = 14;
%     Ttl.BackgroundColor = [1 0 0];

    % check dirs to save exists
    ImageSaveDir =  fullfile(SaveDir, 'Training Images');
    % Make the folder for saving data
    if(~exist(ImageSaveDir,'dir'))
        mkdir(ImageSaveDir);
    end
    
    CoordSaveDir =  fullfile(SaveDir, 'Training Manual Coord');
    % Make the folder for saving data
    if(~exist(ImageSaveDir,'dir'))
        mkdir(ImageSaveDir);
    end

    format shortg
    % calcs patch sizes
    originalSizeX = size(I,2);
    originalSizeY = size(I,1);
    newSize = 150;

    cutoutsINrowX = ceil(originalSizeX/newSize);      % calculate minimum numer of cutouts to produce
    cutoutsINrowY = ceil(originalSizeY/newSize);

    offsetX = (cutoutsINrowX*newSize-originalSizeX)/(cutoutsINrowX-1);

    while offsetX < 20
        cutoutsINrowX = cutoutsINrowX+1;
        offsetX = round((cutoutsINrowX*newSize-originalSizeX)/(cutoutsINrowX-1));      % offset which will be applied for a fix number of cutouts in a row/column
    end

    offsetY = (cutoutsINrowY*newSize-originalSizeY)/(cutoutsINrowY-1);

    while offsetY < 20
        cutoutsINrowY = cutoutsINrowY+1;
        offsetY = round((cutoutsINrowY*newSize-originalSizeY)/(cutoutsINrowY-1));      % offset which will be applied for a fix number of cutouts in a row/column
    end

    numCutouts = cutoutsINrowX*cutoutsINrowY;

    positions_startX = [1 newSize-offsetX:newSize-1-offsetX:((newSize-1)*cutoutsINrowX-offsetX*cutoutsINrowX)];
    positions_endX = positions_startX+(newSize-1);

    positions_startY = [1 newSize-offsetY:newSize-1-offsetY:((newSize-1)*cutoutsINrowY-offsetY*cutoutsINrowY)];
    positions_endY = positions_startY+(newSize-1);

    h = waitbar(0,'Generate Data...');

    % for each patch
    for y_cutout = 1:cutoutsINrowY
        for x_cutout = 1:cutoutsINrowX

            % create patch of 150x150 pixel
            x_start = positions_startX(x_cutout);
            y_start = positions_startY(y_cutout);
            x_end = positions_endX(x_cutout);
            y_end = positions_endY(y_cutout);

            cutout150 = I(y_start:y_end,x_start:x_end);
            currentNum = (y_cutout - 1) * cutoutsINrowX + x_cutout;

            % get conelocs inside patch
            x_rect = [x_start x_end x_end x_start x_start];
            y_rect = [y_start y_start y_end y_end y_start];
            in = inpolygon(conelocs(:,1), conelocs(:,2), x_rect, y_rect);

            name = [baseFileNameNoExt '_' num2str(currentNum) tiff_ext];
            name = fullfile(ImageSaveDir, name);
            % save image
            imwrite(cutout150, name, 'tiff');
            % move points
            roiX = conelocs(:,1);
            roiX = roiX(in);
            roiX = roiX - x_start;

            roiY = conelocs(:,2);
            roiY = roiY(in);
            roiY = roiY - y_start;
            
            % save points
            name = [baseFileNameNoExt '_' num2str(currentNum) '.txt'];
            name = fullfile(CoordSaveDir, name);
            fileID = fopen(name,'w');
            formatSpec = '%4.2f %4.2f\n';
            points = [roiX roiY]';
            fprintf(fileID, formatSpec,points);
            fclose(fileID);

            waitbar(currentNum / (numCutouts))
        end
    end

    close(h)
end
