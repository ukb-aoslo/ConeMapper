function scaleCoefficient = GetScaleCoefficient(unit, pixelsPerDegree)
    % GetScaleCoefficient Returns scale coefficient, which is necessary to 
    % apply to pixel value based on pixels per degree and given unit
    %
    %   - unit - name of the unit (pixel, arcsec, arcmin, degree)
    %   - pixelsPerDegree - number of pixels per one degree of visual angle
    %
    % Returns:
    %   - scaleCoefficient - coefficient which is necessary to multiply on pixel
    %   value

    % 1 degree = 60 arcmin;
    % 1 arcmin = 60 arcsec;
    switch unit
        case 'pixel'
            scaleCoefficient = 1;

        case 'arcsec'
            scaleCoefficient = 60 * 60 / pixelsPerDegree;

        case 'arcmin'
            scaleCoefficient = 60 / pixelsPerDegree;

        case 'degree'
            scaleCoefficient = 1 / pixelsPerDegree;

        otherwise
            error(["Unknown unit: ", unit]);
    end
end

