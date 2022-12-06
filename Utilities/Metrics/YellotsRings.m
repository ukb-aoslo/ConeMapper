classdef YellotsRings < DensityMetricBase
    %YELLOTSRINGS calculates the cone density using DFT and Yellots Rings.
    %For more info, read: https://tvst.arvojournals.org/article.aspx?articleid=2753106
    %Original code: https://github.com/OCVL/Full-Auto-Density-Mapping
    
    properties
        SourceImage = [];
        ROI_size = 150;
        DensityMatrixBeforeSmoothing = [];
        avgPixelSpac = [];
        interpedSpacMap = [];
        interpedConfMap = [];
        sumMap = [];
        imBox = [];
        
        % Fields defined in base class
%         ImageHeight = 0;
%         ImageWidth = 0;
%         DensityMatrix = [];
        
        % for PCD
%         PCD_cppa = [];
%         MinDensity_cppa = 0;
%         PCD_loc = [];
        
        % for CDC
%         CDC20_density = 0;
%         CDC20_loc = [];
%         Stats2 = [];
    end
    
    methods
        function obj = YellotsRings(sourceImage, roiSize)
            %YELLOTSRINGS Construct an instance of this class
            if nargin > 0
              	obj.SourceImage = sourceImage;
            else
                return;
            end

            if nargin > 1
                obj.ROI_size = roiSize;
            end
            
            Recalculate(obj);
        end
        
        function Recalculate(obj)
        %   Recalculate(obj) 
        %   recalculates all the data for density map.
        %   - obj - the current class object.
            
            [obj.DensityMatrix, obj.DensityMatrixBeforeSmoothing, obj.avgPixelSpac, obj.interpedSpacMap, ...
                obj.interpedConfMap, obj.sumMap, obj.imBox] = ...
                YellotsRings.GetDensityMatrix(obj.SourceImage, obj.ROI_size);
            
            [obj.PCD_cppa, obj.MinDensity_cppa, obj.PCD_loc] = YellotsRings.GetMinMaxCPPA(obj.DensityMatrix);
            
            [obj.CDC20_density, obj.CDC20_loc, obj.Stats2] = YellotsRings.GetCDC(obj.PCD_cppa, obj.DensityMatrix);
        end
        
        function s = saveobj(obj)
            % for density map calculation
            s.SourceImage = obj.SourceImage;
            s.DensityMatrix = obj.DensityMatrix;
            s.ROI_size = obj.ROI_size;
            s.DensityMatrixBeforeSmoothing = obj.DensityMatrixBeforeSmoothing;
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
        end
    end
    
    methods(Static)
        function obj = loadobj(s)
            if isstruct(s)
                newObj = YellotsRings(); 
                % for density map calculation
                newObj.SourceImage = s.SourceImage;
                newObj.DensityMatrix = s.DensityMatrix;
                if isfield(s,'ROI_size')
                    newObj.ROI_size = s.ROI_size;
                end
                if isfield(s,'DensityMatrixBeforeSmoothing')
                    newObj.DensityMatrixBeforeSmoothing = s.DensityMatrixBeforeSmoothing;
                end

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

                obj = newObj;
            else
                obj = s;
            end
        end
        
        function [densityMatrix, densityMatrixBeforeSmoothing, avgPixelSpac, blendedImage, blendedConfMap, blendedSumMap, imBox] = ...
                GetDensityMatrix(sourceImage, roiSize)
        %   densityMatrix = GetDensityMatrix(sourceImage, roiSize)
        %   returns a density matrix.
        %   - sourceImage - the source image of retina.
        %   - roiSize - size of the scaning window of the algorithm(?)
        %
        % Returns:
        %   - densityMatrix - the density matrix smoothed by gauss filter.
        %   - densityMatrixBeforeSmoothing - the density matrix before smoothing.
        %   - avgPixelSpac - the average spacing of the image.
        %   - blendedImage - the spacing map of the input image (in pixel spacing).
        %   - blendedConfMap - the confidence map. Mask for density map, which
        %   represents the confidence of the calculated values in density map.
        %   - blendedSumMap - the map corresponding to the amount of ROI overlap across the output map.
        %   - imBox - the bounding region of valid (nonzero, NaN, or Inf) pixels.
        
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
            
            blendedImage = blendedImage ./ blendedConfMap;
            blendedConfMap = blendedConfMap ./ blendedSumMap;
            
            % To density, assuming perfect packing
            densityMatrixBeforeSmoothing = sqrt(3)./ (2*(blendedImage).^2);

            densityMatrix = imgaussfilt(densityMatrixBeforeSmoothing, 8);
        end
    end
end

