classdef EuclidianNCones < handle
    %EuclidianNCones calculates the cone density using the Voronoi patch areas of
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
        GoodPointsMap = [];
    end
    
    methods
        function obj = EuclidianNCones(coneLocs, imageSize, numOfNerestCones, sourceImage)
            %EuclidianNCones Construct an instance of this class
            if nargin > 0
                obj.Vorocones = coneLocs;
            else
                return;
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
                        
            if nargin < 4
                sourceImage = [];
            end
            
            Recalculate(obj, sourceImage);
        end
        
        function Recalculate(obj, sourceImage)
        %   Recalculate(obj, sourceImage) 
        %   recalculates all the data for density map.
        %   - obj - the current class object.
        %   - sourceImage - image of retina.
        
            if isempty(obj.Vorocones) || obj.ImageHeight == 0 || obj.ImageWidth == 0
                error("Invalid JennyDinsity class. No data to calculate density");
            end
            
            obj.ConeAreas = EuclidianNCones.GetConeAreas(obj.Vorocones);
            
            if nargin < 2
                sourceImage = [];
            end
            
            [obj.GoodPointsMap, obj.GoodPointsEdge, obj.DensityMatrix] = ...
                EuclidianNCones.GetDensityMatrix(obj.Vorocones, obj.ImageHeight, obj.ImageWidth, ...
                obj.NumOfNearestCones, obj.ConeAreas, sourceImage);
            
            [obj.PCD_cppa, obj.MinDensity_cppa, obj.PCD_loc] = EuclidianNCones.GetMinMaxCPPA(obj.DensityMatrix);
            
            [obj.CDC20_density, obj.CDC20_loc, obj.Stats2] = EuclidianNCones.GetCDC(obj.PCD_cppa, obj.DensityMatrix);
        end
        
        function s = saveobj(obj)
            % for density map calculation
            s.Vorocones = obj.Vorocones;
            s.ImageHeight = obj.ImageHeight;
            s.ImageWidth = obj.ImageWidth;
            s.NumOfNearestCones = obj.NumOfNearestCones;
            s.DensityMatrix = obj.DensityMatrix;

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
            s.GoodPointsMap = obj.GoodPointsMap;
        end
    end
    
    methods(Static)
        function obj = loadobj(s)
            if isstruct(s)
                newObj = EuclidianNCones(); 
                % for density map calculation
                newObj.Vorocones = s.Vorocones;
                newObj.ImageHeight = s.ImageHeight;
                newObj.ImageWidth = s.ImageWidth;
                newObj.NumOfNearestCones = s.NumOfNearestCones;
                newObj.DensityMatrix = s.DensityMatrix;

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
                newObj.GoodPointsMap = s.GoodPointsMap;
                obj = newObj;
            else
                obj = s;
            end
        end
        
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

        function [goodPointsMap, goodPointsEdge, densityMatrix] = GetDensityMatrix(conelocs, imageHeight, imageWidth, ...
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
            goodPointsEdge = EuclidianNCones.FindMapEdgeByConelocs(goodPointsMap, conelocs);
            
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
        
        function contourCoordinates = GetDensityPercentageContour(densityMatrix, PCD_cppa, percentage)
        %   contourCoordinates = GetDensityPercentageContour(densityMatrix, PCD_cppa)
        %   returns coordinates of the contour for given percentage from
        %   PCD_cppa value.
        %    - densityMatrix -  cone density matrix.
        %    - PCD_cppa - peak cone density.
        %    - percentage - percent starting from PCD_cppa in 0..1.
        %
        %    - contourCoordinates - coordinates of the contour in the same
        %    format as output of matlab "contour" function.
        %    https://nl.mathworks.com/help/matlab/ref/contour.html#f19-795863_sep_mw_d9e727e2-79e4-4cf6-bfaf-431a164d82b0
        %
        %   How to extract all contours:
        %   index = 1;
        %   contours = {};
        %   while index < length(contourCoordinates)
        %       numOfPoints = contourCoordinates(2, index);
        %       contours{end + 1} = contourCoordinates(index+1:index + numOfPoints, :)';
        %       index = index + numOfPoints + 1;
        %   end
        %
        %   Each cell in contours will be 2 column array. Each row will be
        %   point of the polygon edge.
        
            Perc20dens = (1 - percentage) * PCD_cppa;
            
            tempFig = figure;
            tempAx = axes;
            % make contour
            contourCoordinates = contour(tempAx, densityMatrix, [Perc20dens, Perc20dens]);
            delete(tempFig);
        end
        
        function boundaryConelocs = FindMapEdgeByConelocs(BWMap, conelocs)
        %   boundaryConelocs = FindMapEdgeByConelocs(BWMap, conelocs)
        %   returns boundary based on conelocs of a map.
        %   - BWmap - the back-white map.
        %   - conelocs - cone locations on BWMap
            
            % round conelocs to use them as indices
            conelocs = round(conelocs);
            [rows, cols, ~] = size(BWMap);
            conelocs(conelocs(:, 2) > rows, :) = [];
            conelocs(conelocs(:, 1) > cols, :) = [];
            
            % convert them to linear indicies
            indexes = sub2ind([rows, cols], conelocs(:, 2), conelocs(:, 1));
            % mark points where cones are placed on actual map
            onesLogicArray = BWMap(indexes) == 1;
            BWMap(indexes(onesLogicArray)) = 2;
            conesInsideMap = find(BWMap == 2);
            % get normal coordinates of that cones
            [row,col] = ind2sub([rows, cols], conesInsideMap);
            
            % find a boundary of the set
            boundingPoly = boundary(col, row);
            boundaryConelocs = [col(boundingPoly), row(boundingPoly)];
        end
    end
end