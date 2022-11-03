function [densityRadialAverage, radiuses, isApproximated] = GetDensityRadialAverage(densityMatrix, coneDensityCentroid)
    % GetDensityRadialAverage returns density radial average starting in CDC point.
    % Based on Julius Ameln approach.
    %    - densityMatrix - matrix of density values. Matrix is the same size as the
    %    original image.
    %    - coneDensityCentroid - cone density centroid coordinates.
    %
    % Returns:
    %    - densityRadialAverage - density radial average from CDC to the sides.
    %    - radiuses - distances from CDC for each density radial average value.
    %    - isApproximated - logical array. 1 - if NaN is in the calculation or
    %    distance is greater than the distance to the closest side of the image.

    % find all relevant pixel in the matrix (not NaNs)
    [coordY, coordX] = find(~isnan(densityMatrix));
    coordinatesToAllPoints = [coordX,coordY];

    % center all matrix point on the cdc (cdc coord == 0/0)
    coordCDC_X = coordX - coneDensityCentroid(1); 
    coordCDC_Y = coneDensityCentroid(2) - coordY;

    % calculate angle and distance towards cdc for every point
    [~, radiuses] = cart2pol(coordCDC_X, coordCDC_Y);
    % round the distance to next whole number
    radiuses = round(radiuses);

    % get distance to closes side
    [dmHeight, dmWidth, ~] = size(densityMatrix);
    minDistanceToSide = min([coneDensityCentroid(1), coneDensityCentroid(2), dmHeight - coneDensityCentroid(2), dmWidth - coneDensityCentroid(1)]);

    % get coordinates by distance
    coordinatesByDistance = arrayfun(@(a) coordinatesToAllPoints(radiuses == a,:), 0:max(radiuses), 'UniformOutput', false);

    % preallocate memory
    densityRadialAverage = nan(length(coordinatesByDistance), 1);
    isApproximated = false(length(coordinatesByDistance), 1);

    radiuses = unique(radiuses);
     
    % for each distance value
    for iCoord = 1:length(coordinatesByDistance)
        % get density values
        indexTemp = sub2ind([dmHeight, dmWidth], coordinatesByDistance{iCoord}(:,2), coordinatesByDistance{iCoord}(:,1));
        densityValuesToAverage = densityMatrix(indexTemp);
        
        isNaN = isnan(densityValuesToAverage);
        % calculate mean
        densityRadialAverage(iCoord) = mean(densityValuesToAverage(~isNaN));
        % check if it is approximate
        isApproximated(iCoord) = any(isNaN) || radiuses(iCoord) > minDistanceToSide;
    end
end