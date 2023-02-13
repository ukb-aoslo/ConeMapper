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

        case 'millimeter'
            unitString = 'mm';
            
        case 'micrometer'
            unitString = 'µm';

        otherwise
            error(["Unknown unit: ", unit]);
    end
end