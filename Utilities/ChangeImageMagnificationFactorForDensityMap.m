function newMap = ChangeImageMagnificationFactorForDensityMap(map, currentIMF, targetIMF, currentRMF, targetRMF)
% Converts given density map from current IMF to target IMF
%
% INPUT:
% map        - 2d density map in pixels
% currentIMF - current image magnification factor in pixels per degree
% targetIMF  - current image magnification factor in pixels per degree
%
% OUTPUT:
% newMap     - 2d density map in pixels

    % convert to cones/degrees^2 with currentIMF
    scaleCoefficientC = GetScaleCoefficient('degree', currentIMF, currentRMF);
    scaleCoefficientCSq = scaleCoefficientC^2;
    newMap = map ./ (scaleCoefficientCSq);

    % resize map
    newMap = imresize(newMap, (targetIMF/ targetRMF ) / (currentIMF / currentRMF ));

    % convert to pixel with targetIMF
    scaleCoefficientT = GetScaleCoefficient('degree', targetIMF, targetRMF);
    scaleCoefficientTSq = scaleCoefficientT^2;
    newMap = newMap .* (scaleCoefficientTSq);
end

