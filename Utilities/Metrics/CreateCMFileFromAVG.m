load("AverageDensityEuclideanNConesMap.mat")

filename = "averageRightCM.mat";
CreateCMFile(filename, averageMapRight, imageMagnificationFactorRight, retinalMagnificationFactorRight)
filename = "averageLeftCM.mat";
CreateCMFile(filename, averageMapLeft, imageMagnificationFactorLeft, retinalMagnificationFactorLeft)

function CreateCMFile(filename, map, ppd, rmf)
    conecount = 0;
    [imH, imW, ~] = size(map);
    I = zeros(imH, imW);
    conelocs = zeros(0, 3);
    eyeType = 'right';
    rodlocs = [];
    nnd_mean = [];
    yellotsRings = [];
    pixelPerDegree = ppd;
    RMF = rmf;
    [PCD_cppa, minDensity_cppa, PCD_loc] = DensityMetricBase.GetMinMaxCPPA(map);
    [CDC20_density, CDC20_loc, stats2] = DensityMetricBase.GetCDC(PCD_cppa, map);
    
    BWMap = map > 0;
    [row,col] = ind2sub([imH, imW], find(BWMap == 1));
    % find a boundary of the set
    boundingPoly = boundary(col, row, 0.5);
    boundaryConelocs = [col(boundingPoly), row(boundingPoly)];
    
    euclideanNCones = struct(...
        'Vorocones', [], ...
        'ImageHeight', imH, ...
        'ImageWidth', imW, ...
        'NumOfNearestCones', 150, ...
        'DensityMatrix', map, ...
        'PCD_cppa', PCD_cppa, ...
        'MinDensity_cppa', minDensity_cppa, ...
        'PCD_loc', PCD_loc, ...
        'CDC20_density', CDC20_density, ...
        'CDC20_loc', CDC20_loc, ...
        'Stats2', stats2, ...
        'GoodPointsMap', BWMap, ...
        'GoodPointsEdge', boundaryConelocs);
    
    
    save(filename, "euclideanNCones", "yellotsRings", "nnd_mean", "rodlocs" ,"eyeType", "conelocs", "I", "conecount","pixelPerDegree", "RMF");
end