classdef (Abstract) DensityMetricBase < handle
    %DENSITYMETRICBASE Base class for all density classes
    
    properties
        % for density map calculation
        ImageHeight = 0;
        ImageWidth = 0;
        DensityMatrix = [];
        
        % for PCD
        PCD_cppa = [];
        MinDensity_cppa = 0;
        PCD_loc = [];
        
        % for CDC
        CDC20_density = 0;
        CDC20_loc = [];
        Stats2 = [];
    end
    
    methods (Abstract)
        Recalculate(obj);
        saveobj(obj);
    end

    methods (Static, Abstract)
        loadobj(s);
        GetDensityMatrix(conelocsOrImage);
        [PCD_cppa, minDensity_cppa, PCD_loc] = GetMinMaxCPPA(densityMatrix);
        [CDC20_density, CDC20_loc, stats2] = GetCDC(PCD_cppa, densityMatrix);
    end
end

