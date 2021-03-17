function [minCenterB, maxCenterB, avgCenterB, minCornerB, maxCornerB, avgCornerB] = get_brightness_stats(filepath)
    % Fname = 'BAK8044R_2019_04_03_10_31_23_AOSLO_V003_stabilized_840_annotated';
    
    [folder, baseFileNameNoExt, ~] = fileparts(filepath);
    Fname = [folder '\' baseFileNameNoExt];
    mat_ext = '.mat';
    I = 0;
    conelocs = 0;
    
    % load data
    load([Fname mat_ext]);
    conelocs = unique(conelocs, 'rows', 'stable');
    boxposition;
    
    if multiple_mosaics == 1
        try
            ChangeMosaic = I;
            I = ChangeMosaic{1};
        catch
            ChangeMosaic = squeeze(num2cell(I,[1 2]));
            I = ChangeMosaic{1};
        end
    end
    
    imageSize = size(I);
    % I has transposed coordinates
    imageSize = imageSize(:,[2 1]);
    imageSizeTL = [imageSize(1) * 0.36,  imageSize(2) * 0.36];
    imageSizeBR = [imageSize(1) * 0.64,  imageSize(2) * 0.64];
    
    conelocs(:, 3) = 0;
    conelocs(conelocs(:,1)>imageSizeTL(1) & conelocs(:,1)<=imageSizeBR(1)...
        & conelocs(:,2)>imageSizeTL(2) & conelocs(:,2)<=imageSizeBR(2),3)= 1;
    % Calculatiung center
    idx = conelocs(:, 3) == 1;
    newlocs = conelocs(idx, :);
    roundedLocs = round(newlocs(:, 1:2));
    % I has transposed coordinates
    roundedLocs = roundedLocs(:,[2 1]);
    lineaerIndexes = sub2ind(imageSize, roundedLocs(:, 1), roundedLocs(:, 2));
    brighnessValues = I(lineaerIndexes);

    minCenterB = min(brighnessValues);
    maxCenterB = max(brighnessValues);
    avgCenterB = mean(brighnessValues);

    % Calculatiung corner
    idxCorner = conelocs(:, 3) == 0;

    newlocs = conelocs(idxCorner, :);
    roundedLocs = round(newlocs(:, 1:2));
    % I has transposed coordinates
    roundedLocs = roundedLocs(:,[2 1]);
    idxRounded = roundedLocs(:, 1)> 0 & roundedLocs(:, 2)> 0 & ...
                 roundedLocs(:, 1)<= imageSize(1) & roundedLocs(:, 2)<= imageSize(2);
    roundedLocs = roundedLocs(idxRounded, :);
    lineaerIndexes = sub2ind(imageSize, roundedLocs(:, 1), roundedLocs(:, 2));
    brighnessValues = I(lineaerIndexes);
    brighnessValues = brighnessValues(brighnessValues > 0);
    
    minCornerB = min(brighnessValues);
    maxCornerB = max(brighnessValues);
    avgCornerB = mean(brighnessValues);

    if isfloat(minCenterB) && minCenterB <= 1
        minCenterB = minCenterB * 255;
        maxCenterB = maxCenterB * 255;
        avgCenterB = avgCenterB * 255;
        minCornerB = minCornerB * 255;
        maxCornerB = maxCornerB * 255;
        avgCornerB = avgCornerB * 255;
    end
end