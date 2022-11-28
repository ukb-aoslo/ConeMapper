function densityRadialDataStruct = GetDensityRadialAverage(densityMatrix, coneDensityCentroid, percents)
    % GetDensityRadialAverage returns density radial average starting in CDC point.
    % Based on Julius Ameln approach.
    %    - densityMatrix - matrix of density values. Matrix is the same size as the
    %    original image.
    %    - coneDensityCentroid - cone density centroid coordinates.
    %
    % Returns:
    %    - densityRadialDataStruct - the struct with density radial average values.
    %    Containts the next fields:
    %       - densityRadialAverageAll - density radial average for the full circle.
    %       - densityRadialStdAll - 
    %       - radiusesAll - distances from CDC for each density radial average value.
    %       - densityRadialAverageHorizontal - density radial average for the
    %       top and bottom 10 degrees sectors.
    %       - densityRadialStdHorizontal - 
    %       - radiusesHorizontal - distances from CDC for each density radial average
    %       value  for the top and bottom 10 degrees sectors.
    %       - densityRadialAverageVertical - density radial average for the right and
    %       left 10 degrees sectors.
    %       - densityRadialStdVertical - 
    %       - radiusesVertical - distances from CDC for each density radial average value the right and
    %       left 10 degrees sectors.
    %       - isApproximated - logical array. 1 - if NaN is in the calculation or
    %    distance is greater than the distance to the closest side of the image.

    if ~exist("percents", "var")
        percents = [10, 90];
    else
        percents = sort(percents);
    end

    % find all relevant pixel in the matrix (not NaNs)
    [dmHeight, dmWidth, ~] = size(densityMatrix);
    [XX,YY] = ndgrid(1:dmWidth, 1:dmHeight);
    coordX = XX(:);
    coordY = YY(:);
%     [coordY, coordX] = find(~isnan(densityMatrix));
    coordinatesToAllPoints = [coordX,coordY];

    % center all matrix point on the cdc (cdc coord == 0/0)
    coordCDC_X = coordX - coneDensityCentroid(1); 
    coordCDC_Y = coneDensityCentroid(2) - coordY;

    % calculate angle and distance towards cdc for every point
    [angles, radiuses] = cart2pol(coordCDC_X, coordCDC_Y);
    % round the distance to next whole number
    radiuses = round(radiuses);

    % transfer radians to degree
    angles = rad2deg(angles); 
    % switch negative degree to full circle degree
    angles(angles < 0) = 360 + angles(angles < 0); 

    horizontalLine = (angles <= 185 & angles >= 175) | (angles <= 5 | angles >= 355) | radiuses == 0;
    verticalLine = (angles <= 95 & angles >= 85) | (angles <= 275 & angles >= 265) | radiuses == 0;

    densityRadialDataStruct = struct(...
        'densityRadialAverageAll', [], 'densityRadialStdAll', [], 'radiusesAll', [], ...
        'densityRadialAverageHorizontal', [], 'densityRadialStdHorizontal', [], 'radiusesHorizontal', [], ...
        'densityRadialAverageVertical', [], 'densityRadialStdVertical', [], 'radiusesVertical', [], ...
        'isApproximated', [], 'densityRadialPercentiles', [], 'percents', []);
    [densityRadialDataStruct.densityRadialAverageAll,...
        densityRadialDataStruct.densityRadialStdAll,...
        densityRadialDataStruct.isApproximated,...
        densityRadialDataStruct.radiusesAll,...
        densityRadialDataStruct.densityRadialPercentiles] =...
        CalculateRadialAvg(densityMatrix, coordinatesToAllPoints, radiuses, coneDensityCentroid, percents);
    [densityRadialDataStruct.densityRadialAverageHorizontal,...
        densityRadialDataStruct.densityRadialStdHorizontal,...
        ~, ...
        densityRadialDataStruct.radiusesHorizontal, ~] =...
        CalculateRadialAvg(densityMatrix, coordinatesToAllPoints(horizontalLine, :), radiuses(horizontalLine), coneDensityCentroid, percents);
    [densityRadialDataStruct.densityRadialAverageVertical,...
        densityRadialDataStruct.densityRadialStdVertical,...
        ~, ...
        densityRadialDataStruct.radiusesVertical, ~] =...
        CalculateRadialAvg(densityMatrix, coordinatesToAllPoints(verticalLine, :), radiuses(verticalLine), coneDensityCentroid, percents);
    densityRadialDataStruct.percents = percents;
end

function [densityRadialAverage, densityRadialStd, isApproximated, radiuses, densityRadialPercentiles] = ...
    CalculateRadialAvg(densityMatrix, coordinateToPoints, radiuses, coneDensityCentroid, percents)

    % get distance to closes side
    [dmHeight, dmWidth, ~] = size(densityMatrix);
    minDistanceToSide = min([coneDensityCentroid(1), coneDensityCentroid(2), dmHeight - coneDensityCentroid(2), dmWidth - coneDensityCentroid(1)]);

    % get coordinates by distance
    coordinatesByDistance = arrayfun(@(a) coordinateToPoints(radiuses == a,:), 0:max(radiuses), 'UniformOutput', false);

    % preallocate memory
    densityRadialAverage = nan(length(coordinatesByDistance), 1);
    densityRadialStd = nan(length(coordinatesByDistance), 1);
    isApproximated = false(length(coordinatesByDistance), 1);
    densityRadialPercentiles = nan(length(percents), length(coordinatesByDistance));

    radiuses = unique(radiuses);
     
    % for each distance value
    for iCoord = 1:length(coordinatesByDistance)
        % get density values
        indexTemp = sub2ind([dmHeight, dmWidth], coordinatesByDistance{iCoord}(:,2), coordinatesByDistance{iCoord}(:,1));
        densityValuesToAverage = densityMatrix(indexTemp);
        
        isNaN = isnan(densityValuesToAverage);
        % calculate mean
        densityRadialAverage(iCoord) = mean(densityValuesToAverage(~isNaN));
        densityRadialStd(iCoord) = std(densityValuesToAverage(~isNaN));

        densityRadialPercentiles(:, iCoord) = prctile(densityValuesToAverage(~isNaN), percents);
        % check if it is approximate
        isApproximated(iCoord) = any(isNaN) || radiuses(iCoord) > minDistanceToSide;
    end

    firstNanIndex = find(isnan(densityRadialAverage), 1, 'first');
    if firstNanIndex > 1
        densityRadialAverage = densityRadialAverage(1:firstNanIndex-1);
        densityRadialStd = densityRadialStd(1:firstNanIndex-1);
        isApproximated = isApproximated(1:firstNanIndex-1);
        radiuses = radiuses(1:firstNanIndex-1);
        densityRadialPercentiles = densityRadialPercentiles(:, 1:firstNanIndex-1);
    end
end