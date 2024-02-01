function [newImage] = CreateZScoreMap(densityMap, avgDensMap, stdMap, cdcPoint)
    edgeStep = 30;
    si = size(avgDensMap);
    mean_center(1) = floor(si(2)/2)+1;
    mean_center(2) = floor(si(1)/2)+1;

    [rsdm, csdm] = size(densityMap(edgeStep+1:size(densityMap,1)-edgeStep, edgeStep+1:size(densityMap,2)-edgeStep));
    center = cdcPoint - edgeStep;

    diffWidthToZero = mean_center(1) - center(1);
    diffHeightToZero = mean_center(2) - center(2);
    diffWidthToLimit = (si(2) - mean_center(1)) - (csdm - center(1));
    diffHeightToLimit = (si(1) - mean_center(2)) - (rsdm - center(2));

    % Update avgDensMap size if necessary
    if diffWidthToZero < 0 ...
        || diffHeightToZero < 0 ...
        || diffWidthToLimit < 0 ...
        || diffHeightToLimit < 0
        
        widthAds = 0;
        heighAds = 0;

        if diffWidthToZero < 0
            widthAds = -diffWidthToZero;
        end

        if diffHeightToZero < 0
            heighAds = -diffHeightToZero;
        end

        if diffWidthToLimit < 0
            widthAds = widthAds - diffWidthToLimit;
        end

        if diffHeightToLimit < 0
            heighAds = heighAds - diffHeightToLimit;
        end

        if rem(widthAds, 2) ~= 0
            widthAds = widthAds + 1;
        end

        if rem(heighAds, 2) ~= 0
            heighAds = heighAds + 1;
        end

        newMap = zeros(si(1) + heighAds, si(2) + widthAds);

        newMap(heighAds/2+1:end-heighAds/2, widthAds/2+1:end-widthAds/2) = avgDensMap;
        avgDensMap = newMap;
        si = size(avgDensMap);
        mean_center(1) = floor(si(2)/2)+1;
        mean_center(2) = floor(si(1)/2)+1;
    end


    % Compute mean matrix %%% just include normal matrix
    rc_start = round(mean_center-center);
    % Compute place of personal density matrix within mean matrix area
    density_matrix_plac_temp = NaN(si);
    density_matrix_plac_temp(rc_start(2)+1:rc_start(2)+rsdm,rc_start(1)+1:rc_start(1)+csdm) = ...
        densityMap(edgeStep+1:size(densityMap,1)-edgeStep, edgeStep+1:size(densityMap,2)-edgeStep);


    % calculate z-score
    avgDensMap = (density_matrix_plac_temp - avgDensMap) ./ stdMap ;
    newImage = avgDensMap(rc_start(2)-edgeStep+1:rc_start(2)+rsdm+edgeStep,rc_start(1)-edgeStep+1:rc_start(1)+csdm+edgeStep);
end

