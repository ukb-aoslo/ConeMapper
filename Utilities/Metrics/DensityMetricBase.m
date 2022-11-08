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
    end

    methods (Static)
        function [PCD_cppa, minDensity_cppa, PCD_loc] = GetMinMaxCPPA(densityMatrix)
        %   [PCD_cppa, minDensity_cppa, PCD_loc] = GetMinMaxCPPA(densityMatrix)
        %   returns peak cone density and the supplementary information
        %
        %    - densityMatrix - the density matrix
        % Returns:
        %    - PCD_cppa - peak cone density (PCD) value
        %    - minDensity_cppa - minimum density value
        %    - PCD_loc - coordinates of PCD in the density matrix

            minDensity_cppa = min(densityMatrix(:));

            [maxValues, rowMaxIndexes] = max(densityMatrix);
            [PCD_cppa, maxValueColIndex] = max(maxValues);
            maxValueRowIndex = rowMaxIndexes(maxValueColIndex);
            PCD_loc = [maxValueColIndex, maxValueRowIndex];
        end

        function [CDC20_density, CDC20_loc, stats2] = GetCDC(PCD_cppa, densityMatrix)
        %   [CDC20_density, CDC20_loc, stats2] = GetCDC(PCD_cppa, densityMatrix)
        %   returns cone density centroid and the supplementary information.
        %   - PCD_cppa - peak cone density.
        %   - densityMatrix -  cone density matrix.
        %
        % Returns:
        %    - CDC20_density - CDC density value
        %    - CDC20_loc - CDC location
        %    - stats2 - raw measured statistic structure

            Perc20dens = 0.8 * PCD_cppa;
            
            density_plot_norm = mat2gray(densityMatrix);
            L = zeros(size(densityMatrix));
            L(densityMatrix > Perc20dens) = ones;
            stats2 = regionprops(L,density_plot_norm, 'WeightedCentroid');

            CDC20_loc = [round(stats2.WeightedCentroid(1)), round(stats2.WeightedCentroid(2))];
            CDC20_density = densityMatrix(round(CDC20_loc(2)), round(CDC20_loc(1)));
        end
    end
end

