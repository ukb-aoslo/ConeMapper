function unitString = GetUnitsToPrint(unit)
% GetUnitsToPrint returns printable abbreviations of the given unit
    switch unit
        case 'pixel'
            unitString = 'px';

        case 'arcsec'
            unitString = 'arcsec';

        case 'arcmin'
            unitString = 'arcmin';

        case 'degree'
            unitString = 'deg';

        otherwise
            error(["Unknown unit: ", unit]);
    end
end

