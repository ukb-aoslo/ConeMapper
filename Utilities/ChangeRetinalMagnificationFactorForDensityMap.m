function newMap = ChangeRetinalMagnificationFactorForDensityMap(map, currentIMF, currentRMF, targetRMF)
% Converts given density map from current IMF to target IMF
%
% INPUT:
% map        - 2d density map in pixels
% currentIMF - current image magnification factor in pixels per degree
% currentRMF - current retinal magnification factor in µm per degree
% targetRMF  - target retinal magnification factor in µm per degree
%
% OUTPUT:
% newMap     - 2d density map in pixels

% REMARK:
%   Map should be already converted to proper IMF


    % convert to cones/micrometer^2
    scaleCoefficientC = GetScaleCoefficient('micrometer', currentIMF, currentRMF);
    scaleCoefficientCSq = scaleCoefficientC^2;
    newMap = map ./ (scaleCoefficientCSq);

    % resize map
    newMap = imresize(newMap, (currentRMF/ targetRMF ));

    % convert to pixel
    scaleCoefficientT = GetScaleCoefficient('micrometer', currentIMF, targetRMF);
    scaleCoefficientTSq = scaleCoefficientT^2;
    newMap = newMap .* (scaleCoefficientTSq);
end