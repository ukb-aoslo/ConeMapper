classdef EuclideanNCones < DensityMetricBase
    %EuclideanNCones calculates the cone density using the Voronoi patch areas of
    % the k nearest cones to each pixel by Jenny's algorithm
    
    % requested input data
    % - locations of all cones which could be identified in the reference image
    % - reference image size
    
    properties
        % for density map calculation
        Vorocones = [];
        ConeAreas = [];
        NumOfNearestCones = 150;

        % points which represent a polygon inside of which we have non
        % aproximated map
        GoodPointsEdge = [];
        GoodPointsMap = [];

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
        function obj = EuclideanNCones(coneLocs, imageSize, numOfNerestCones, sourceImage)
            %EuclideanNCones Construct an instance of this class
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
            
            obj.ConeAreas = EuclideanNCones.GetConeAreas(obj.Vorocones);
            
            if nargin < 2
                sourceImage = [];
            end
            
            [obj.GoodPointsMap, obj.GoodPointsEdge, obj.DensityMatrix] = ...
                EuclideanNCones.GetDensityMatrix(obj.Vorocones, obj.ImageHeight, obj.ImageWidth, ...
                obj.NumOfNearestCones, obj.ConeAreas, sourceImage);
            
            [obj.PCD_cppa, obj.MinDensity_cppa, obj.PCD_loc] = EuclideanNCones.GetMinMaxCPPA(obj.DensityMatrix);
            
            [obj.CDC20_density, obj.CDC20_loc, obj.Stats2] = EuclideanNCones.GetCDC(obj.PCD_cppa, obj.DensityMatrix);
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
                newObj = EuclideanNCones(); 
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
        %   cone and number of neighbour cones for each cone.
        %
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
        %
        %   - conelocs - locations of cones.
        %   - imageHeight - height of the source image.
        %   - imageWidth - width of the source image.
        %   - numOfNearestCones - number of cones in the area.
        %   - coneArea - array with area of each cone.
        %   - sourceImage - the source image. Used to exclude points where we don't
        %   have image data from densityMatrix.
        %
        % Returns:
        %   - densityMatrix - the density matrix (imageHeight * imageWidth).
        %   - goodPointsMap - mask of density matrix, representing a points without
        %   approximation.
        %   - goodPointsEdge - the polygon points inside of which density values are
        %   not approximated.
            
            boundingPoly = boundary(conelocs(:, 1), conelocs(:, 2), 1);
            
            % get starting coordinates (exclude the border area)
            pixelStartX = round(max([min(conelocs(:,1)), 1]));
            pixelEndX   = round(min([max(conelocs(:,1)), imageWidth]));
            pixelStartY = round(max([min(conelocs(:,2)), 1]));
            pixelEndY   = round(min([max(conelocs(:,2)), imageHeight]));

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
                    smallestIdx =  distIDXsorted(1:min([numOfNearestCones, length(distIDXsorted)]));
                    
                    isBoundingPolyPoint = ismember(smallestIdx, boundingPoly);
                    
                    availiableCones = numOfNearestCones;
                    % if there is at list one cone with extremly big area
                    if any(isBoundingPolyPoint)
                        isBoundingPolyPoint = ismember(distIDXsorted, boundingPoly);
                        distIDXsorted = distIDXsorted(~isBoundingPolyPoint);
                        availiableCones = min([numOfNearestCones, length(distIDXsorted)]);
                        smallestIdx =  distIDXsorted(1:availiableCones);
                    else
                        goodPointsMap(coorY, coorX) = 1;
                    end
                    
                    % get voronoi areas of cones
                    areaNearestVoronois = coneArea(smallestIdx);
                    
                    % calculate sum area of selected cones
                    densityAreaVoronois = sum(areaNearestVoronois);
                    % get density as number of cones divided by area they cover
                    % cppa - cones per pixel area
                    densityMatrix(coorY, coorX) = availiableCones / densityAreaVoronois;                    
                end                                        % end of coorX loop

                waitbar((coorY - pixelStartY) / nSteps)
            end                                            % end of coorY loop

            % get the edge of non aproximated area
            goodPointsEdge = EuclideanNCones.FindMapEdgeByConelocs(goodPointsMap, conelocs, 0.5);
            
            % TODO: make conelocsBoundingPoly calc here for black lines
            if ~isempty(sourceImage)
                boundaryConelocs = EuclideanNCones.FindMapEdgeByConelocs(ones(imageHeight, imageWidth), conelocs, 1);
                bw = poly2mask(boundaryConelocs(:, 1), boundaryConelocs(:, 2), imageHeight, imageWidth);
                densityMatrix(sourceImage < 8 & bw == 0) = NaN;
            end
            close(progressBar);
        end
        
        function boundaryConelocs = FindMapEdgeByConelocs(BWMap, conelocs, shrinkFactor)
        %   boundaryConelocs = FindMapEdgeByConelocs(BWMap, conelocs)
        %   returns boundary based on conelocs of a map.
        %   - BWmap - the back-white map (the mask).
        %   - conelocs - cone locations on BWMap
        %
        % Returns:
        %   - boundaryConelocs - the polygon points inside of which we have all
        %   cones, which are placed not on the zeros of the mask
            
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
            
            if isempty(conesInsideMap)
                row = conelocs(:, 2);
                col = conelocs(:, 1);
            end
            
            % find a boundary of the set
            boundingPoly = boundary(col, row, shrinkFactor);
            boundaryConelocs = [col(boundingPoly), row(boundingPoly)];
        end
    end
end