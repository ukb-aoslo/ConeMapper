function scaleCoefficient = GetScaleCoefficient(unit, ImageMagnificationFactor, rmf)
    % GetScaleCoefficient Returns scale coefficient, which is necessary to 
    % apply to pixel value based on pixels per degree and given unit
    %
    %   - unit - name of the unit (pixel, arcsec, arcmin, degree)
    %   - ImageMagnificationFactor - number of pixels per one degree of visual angle
    %
    % Returns:
    %   - scaleCoefficient - coefficient which is necessary to multiply on pixel
    %   value

    % 1 degree = 60 arcmin;
    % 1 arcmin = 60 arcsec;

    if nargin < 3
        rmf = NaN;
    end

    switch unit
        case {'pixel', 'px', ''}
            scaleCoefficient = 1;

        case 'arcsec'
            scaleCoefficient = 60 * 60 / ImageMagnificationFactor;

        case 'arcmin'
            scaleCoefficient = 60 / ImageMagnificationFactor;

        case {'degree', 'deg'}
            scaleCoefficient = 1 / ImageMagnificationFactor;

        % micrometer = 10^(-6) meter
        case {'micrometer', 'µm'}
            scaleCoefficient = rmf / ImageMagnificationFactor;

        % millimeter = 10^(-3) meter
        case {'millimeter', 'mm'}
            scaleCoefficient = 0.001 * rmf / ImageMagnificationFactor;

        otherwise
            error(["Unknown unit: ", unit]);
    end
end

