classdef YellotsRings < handle
    %YELLOTSRINGS calculates the cone density using DFT and Yellots Rings.
    %For more info, read: https://tvst.arvojournals.org/article.aspx?articleid=2753106
    %Original code: https://github.com/OCVL/Full-Auto-Density-Mapping
    
    properties
        SourceImage = [];
        ROI_size = 200;
        DensityMatrix = [];
        avgPixelSpac = [];
        interpedSpacMap = [];
        interpedConfMap = [];
        sumMap = [];
        imBox = [];
        
        % for PCD
        PCD_cppa = [];
        MinDensity_cppa = 0;
        PCD_loc = [];
        
        % for CDC
        CDC20_density = 0;
        CDC20_loc = [];
        Stats2 = [];
        
        % points which represent a polygon inside of which we have non
        % aproximated map
        GoodPointsEdge = [];
    end
    
    methods
        function obj = YellotsRings(sourceImage)
            %YELLOTSRINGS Construct an instance of this class
            if nargin > 0
              	obj.SourceImage = sourceImage;
            else
                return;
            end
            
            Recalculate(obj);
        end
        
        function Recalculate(obj)
        %   Recalculate(obj) 
        %   recalculates all the data for density map.
        %   - obj - the current class object.
            
            [obj.DensityMatrix, obj.avgPixelSpac, obj.interpedSpacMap, ...
                obj.interpedConfMap, obj.sumMap, obj.imBox] = ...
                YellotsRings.GetDensityMatrix(obj.SourceImage, obj.ROI_size);
            
            [obj.PCD_cppa, obj.MinDensity_cppa, obj.PCD_loc] = YellotsRings.GetMinMaxCPPA(obj.DensityMatrix);
            
            [obj.CDC20_density, obj.CDC20_loc, obj.Stats2] = YellotsRings.GetCDC(obj.PCD_cppa, obj.DensityMatrix);
        end
        
        function s = saveobj(obj)
            % for density map calculation
            s.SourceImage = obj.SourceImage;
            s.DensityMatrix = obj.DensityMatrix;
            s.avgPixelSpac = obj.avgPixelSpac;
            s.interpedSpacMap = obj.interpedSpacMap;
            s.interpedConfMap = obj.interpedConfMap;
            s.sumMap = obj.sumMap;
            s.imBox = obj.imBox;

            % for PCD
            s.PCD_cppa = obj.PCD_cppa;
            s.MinDensity_cppa = obj.MinDensity_cppa;
            s.PCD_loc = obj.PCD_loc;

            % for CDC
            s.CDC20_density = obj.CDC20_density;
            s.CDC20_loc = obj.CDC20_loc;
            s.Stats2 = obj.Stats2;

            % points which represent a polygon inside of which we have non
            % aproximated map
            s.GoodPointsEdge = obj.GoodPointsEdge;
        end
    end
    
    methods(Static)
        function obj = loadobj(s)
            if isstruct(s)
                newObj = YellotsRings(); 
                % for density map calculation
                newObj.SourceImage = s.SourceImage;
                newObj.DensityMatrix = s.DensityMatrix;
                newObj.avgPixelSpac = s.avgPixelSpac;
                newObj.interpedSpacMap = s.interpedSpacMap;
                newObj.interpedConfMap = s.interpedConfMap;
                newObj.sumMap = s.sumMap;
                newObj.imBox = s.imBox;
                
                % for PCD
                newObj.PCD_cppa = s.PCD_cppa;
                newObj.MinDensity_cppa = s.MinDensity_cppa;
                newObj.PCD_loc = s.PCD_loc;

                % for CDC
                newObj.CDC20_density = s.CDC20_density;
                newObj.CDC20_loc = s.CDC20_loc;
                newObj.Stats2 = s.Stats2;

                % points which represent a polygon inside of which we have non
                % aproximated map
                newObj.GoodPointsEdge = s.GoodPointsEdge;
                obj = newObj;
            else
                obj = s;
            end
        end
        
        function [densityMatrix, avgPixelSpac, blendedImage, blendedConfMap, blendedSumMap, imBox] = ...
                GetDensityMatrix(sourceImage, roiSize)
        %   densityMatrix = GetDensityMatrix(sourceImage)
        %   returns a density matrix.
        %   - sourceImage - the source image of retina.
        
            [avgPixelSpac, interpedSpacMap, interpedConfMap, sumMap, imBox] = ...
                FitFourierSpacing(sourceImage, roiSize);
            
            [rows, cols, ~] = size(sourceImage);
            blendedImage = zeros(rows, cols);
            blendedConfMap = zeros(rows, cols);
            blendedSumMap = zeros(rows, cols);
            
            if ~isempty(imBox)
                blendedImage( imBox(2):imBox(2)+imBox(4),...
                           imBox(1):imBox(1)+imBox(3) ) = blendedImage( imBox(2):imBox(2)+imBox(4),...
                                                                           imBox(1):imBox(1)+imBox(3) ) + interpedSpacMap;

                blendedSumMap( imBox(2):imBox(2)+imBox(4),...
                         imBox(1):imBox(1)+imBox(3) ) = blendedSumMap( imBox(2):imBox(2)+imBox(4),...
                                                                       imBox(1):imBox(1)+imBox(3) ) + sumMap;

                blendedConfMap( imBox(2):imBox(2)+imBox(4),...
                           imBox(1):imBox(1)+imBox(3) ) = blendedConfMap( imBox(2):imBox(2)+imBox(4),...
                                                                           imBox(1):imBox(1)+imBox(3) ) + interpedConfMap;   
            end
            
            blendedImage = blendedImage ./ blendedSumMap;
            blendedConfMap = blendedConfMap ./ blendedSumMap;
                        
%             % arcmins per pixel
%             scaling = 0.1;
%             % To density, assuming perfect packing
%             densityMatrix = sqrt(3)./ (2*(blendedImage * scaling).^2);
% % 
% %             if strcmp(unit,'microns (mm density)')
% %                 densityMatrix = (1000^2).*densityMatrix;
% %             end
%     
            % TODO: check this step!!!!!!!
            densityMatrix = blendedConfMap ./ blendedImage;
            densityMatrix = imgaussfilt(densityMatrix, 8);
        end
        
        function [PCD_cppa, minDensity_cppa, PCD_loc] = GetMinMaxCPPA(densityMatrix)
        %   [PCD_cppa, minDensity_cppa, PCD_loc] = GetMinMaxCPPA(densityMatrix)
        %   returns peak cone density (PCD), minimum density value, and coordinates of
        %   PCD in the density matrix

            minDensity_cppa = min(densityMatrix(:));

            [maxValues, rowMaxIndexes] = max(densityMatrix);
            [PCD_cppa, maxValueColIndex] = max(maxValues);
            maxValueRowIndex = rowMaxIndexes(maxValueColIndex);
            PCD_loc = [maxValueColIndex, maxValueRowIndex];
        end

        function [CDC20_density, CDC20_loc, stats2] = GetCDC(PCD_cppa, densityMatrix)
        %   [CDC20_density, CDC20_loc, stats2] = GetCDC(PCD_cppa, densityMatrix)
        %   returns CDC density value, CDC location and raw measured statistic
        %   structure.
        %   - PCD_cppa - peak cone density.
        %   - densityMatrix -  cone density matrix.
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

