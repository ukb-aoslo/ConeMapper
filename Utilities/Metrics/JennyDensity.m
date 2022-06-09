classdef JennyDensity < handle
    %JENNYDENSITY calculates the cone density using the Voronoi patch areas of
    % the k nearest cones to each pixel by Jenny's algorithm
    
    % requested input data
    % - locations of all cones which could be identified in the reference image
    % - reference image size
    
    properties
        % for density map calculation
        Vorocones = [];
        ConeAreas = [];
        ImageHeight = 0;
        ImageWidth = 0;
        NumOfNearestCones = 150;
        DensityMatrix = [];
        
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
        function obj = JennyDensity(coneLocs, imageSize, numOfNerestCones, sourceImage)
            %JENNYDENSITY Construct an instance of this class
            %   Detailed explanation goes here
            if nargin > 0
                obj.Vorocones = coneLocs;
            end
            
            if nargin > 1
                obj.ImageHeight = imageSize(1);
                obj.ImageWidth = imageSize(2);
            else
                obj.ImageHeight = max(coneLocs(:, 2));
                obj.ImageWidth = max(coneLocs(:, 1));
            end
            
            if nargin > 2
                obj.NumOfNearestCones = numOfNerestCones;
            end
            
            obj.ConeAreas = JennyDensity.GetConeAreas(obj.Vorocones);
                        
            if nargin < 4
                sourceImage = [];
            end
            
            Recalculate(obj, 0, sourceImage);
        end
        
        function Recalculate(obj, withConeAreas, sourceImage)
        %   Recalculate(obj, withConeAreas) 
        %   recalculates all the data for density map.
        %   - obj - the current class object.
        %   - withConeAreas - the flag that indicates that coneAreas must
        %   be recalculated.
            if withConeAreas
                obj.ConeAreas = JennyDensity.GetConeAreas(obj.Vorocones);
            end
            
            if nargin < 3
                sourceImage = [];
            end
            
            [obj.GoodPointsEdge, obj.DensityMatrix] = ...
                JennyDensity.GetDensityMatrix(obj.Vorocones, obj.ImageHeight, obj.ImageWidth, ...
                obj.NumOfNearestCones, obj.ConeAreas, sourceImage);
            
            [obj.PCD_cppa, obj.MinDensity_cppa, obj.PCD_loc] = JennyDensity.GetMinMaxCPPA(obj.DensityMatrix);
            
            [obj.CDC20_density, obj.CDC20_loc, obj.Stats2] = JennyDensity.GetCDC(obj.PCD_cppa, obj.MinDensity_cppa, obj.DensityMatrix);
        end
        
    end
    
    methods(Static)
        function coneArea = GetConeAreas(vorocones)
        %   coneArea = GetConeAreas(vorocones) returns area of each
        %   cone and number of neighbor cones for each cone.
        %   - vorocones - N*2 vector where first column is X coordinate of N cones,
        %       second column is Y coordinate of N cones.

            dt = delaunayTriangulation(vorocones(:,1),vorocones(:,2));
            [V,C] = voronoiDiagram(dt);
            nVoronoiCells = length ( C );
            coneArea = zeros(nVoronoiCells, 1);
            for cone = 1 : nVoronoiCells
                coneArea(cone) = polyarea(V(C{cone},1),V(C{cone},2));

                % exclude extremely high border values 
                if coneArea(cone) > 1000
                    coneArea(cone) = NaN;
                end
            end
        end

        function [goodPointsEdge, densityMatrix] = GetDensityMatrix(conelocs, imageHeight, imageWidth, ...
            numOfNearestCones, coneArea, sourceImage)
        %   densityMatrix = GetDensityMatrix(conelocs, imageHeight, imageWidth)
        %   returns a density matrix.
        %   - conelocs - locations of cones.
        %   - imageHeight - height of the source image.
        %   - imageWidth - width of the source image.
        %   - numOfNearestCones - number of cones in the area.
        %   - coneArea - array with area of each cone.
            
            boundingPoly = boundary(conelocs(:, 1), conelocs(:, 2), 1);
            
            % get starting coordinates (exclude the border area)
            pixelStartX = round(min(conelocs(:,1)));
            pixelEndX   = round(max(conelocs(:,1)));
            pixelStartY = round(min(conelocs(:,2)));
            pixelEndY   = round(max(conelocs(:,2)));

            % preallocate memory for density matrix
            densityMatrix = nan(imageHeight, imageWidth);
            goodPointsMap = zeros(imageHeight, imageWidth);
            
            % vectors with ROI pixel indexes
            pixelsY = pixelStartY:pixelEndY;
            pixelsX = pixelStartX:pixelEndX;
            % number of steps
            nSteps = length(pixelsY);

            progressBar = waitbar(0, ['ConeDensity analysis - ', num2str(numOfNearestCones),' nearest cones']);

            % for each column
            for coorY = pixelsY                  % central pixel (x) of selection
                % get row coordinates in pixels
                row = combvec(pixelsX, coorY)';
                % calculate distances from pixels in the row to each cone
                distances = pdist2(conelocs, row);
                % sort each column ascending
                [~, originalColumnIndexes] = sort(distances, 1);

                % for each pixel
                for coorX = pixelsX              % central pixel (y) of selection
                    % get indexes of cones with the smallest distances to current
                    % pixel
                    distIDXsorted = originalColumnIndexes(:, coorX - pixelStartX + 1);
                    smallestIdx =  distIDXsorted(1:numOfNearestCones);
                    
                    isBoundingPolyPoint = ismember(smallestIdx, boundingPoly);
                    
                    % if there is at list one cone with extremly big area
                    if any(isBoundingPolyPoint)
                        isBoundingPolyPoint = ismember(distIDXsorted, boundingPoly);
                        distIDXsorted = distIDXsorted(~isBoundingPolyPoint);
                        smallestIdx =  distIDXsorted(1:numOfNearestCones);
                    else
                        goodPointsMap(coorY, coorX) = 1;
                    end
                    
                    % get voronoi areas of cones
                    areaNearestVoronois = coneArea(smallestIdx);
                    
                    % calculate sum area of selected cones
                    densityAreaVoronois = sum(areaNearestVoronois);
                    % get density as number of cones divided by area they cover
                    % cppa - cones per pixel area
                    densityMatrix(coorY, coorX) = numOfNearestCones / densityAreaVoronois;                    
                end                                        % end of coorX loop

                waitbar((coorY - pixelStartY) / nSteps)
            end                                            % end of coorY loop

            % get the edge of non aproximated area
            goodPointsEdge = JennyDensity.FindMapEdgeByConelocs(goodPointsMap, conelocs);
            
            if ~isempty(sourceImage)
                densityMatrix(sourceImage < 8) = NaN;
            end
            close(progressBar);
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

        function [CDC20_density, CDC20_loc, stats2] = GetCDC(PCD_cppa, minDensity_cppa, densityMatrix)
        %   [CDC20_density, CDC20_loc, stats2] = GetCDC(PCD_cppa, minDensity_cppa, densityMatrix)
        %   returns CDC density value, CDC location and raw measured statistic
        %   structure.
        %   - PCD_cppa - peak cone density.
        %   - minDensity_cppa - minimum density value.
        %   - densityMatrix -  cone density matrix.
            densRange = PCD_cppa - minDensity_cppa;
            valuePct = zeros(9);
            for perc = 1:9
                valuePct(perc) = PCD_cppa - densRange * (perc / 10);
            end

            density_plot_norm = mat2gray(densityMatrix);
            L = zeros(size(densityMatrix));
            L(densityMatrix > valuePct(2)) = ones;
            stats2 = regionprops(L,density_plot_norm, 'WeightedCentroid');

            CDC20_loc = [round(stats2.WeightedCentroid(1)), round(stats2.WeightedCentroid(2))];
            CDC20_density = densityMatrix(round(CDC20_loc(2)), round(CDC20_loc(1)));
        end
        
        function boundaryConelocs = FindMapEdgeByConelocs(BWMap, conelocs)
        %   boundaryConelocs = FindMapEdgeByConelocs(BWMap, conelocs)
        %   returns boundary based on conelocs of a map.
        %   - BWmap - the back-white map.
        %   - conelocs - cone locations on BWMap
            
            % round conelocs to use them as indices
            conelocs = round(conelocs);
            [rows, cols, ~] = size(BWMap);
            conelocs(conelocs(:, 1) > rows, :) = [];
            conelocs(conelocs(:, 2) > cols, :) = [];
            
            % convert them to linear indicies
            indexes = sub2ind([rows, cols], conelocs(:, 2), conelocs(:, 1));
            % mark points where cones are placed on actual map
            onesLogicArray = BWMap(indexes) == 1;
            BWMap(indexes(onesLogicArray)) = 2;
            conesInsideMap = find(BWMap == 2);
            % get normal coordinates of that cones
            [row,col] = ind2sub([rows, cols], conesInsideMap);
            
            % find a boundary of the set
            boundingPoly = boundary(row, col);
            boundaryConelocs = [row(boundingPoly), col(boundingPoly)];
        end
    end
end

