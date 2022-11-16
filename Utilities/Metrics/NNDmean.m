classdef NNDmean < DensityMetricBase
    %NNDmean calculates the cone density using average of the
    %distances to the neighbor cones. Neighbor in terms of the Voronoi
    %diagram
    
    % requested input data
    % - locations of all cones which could be identified in the reference image
    % - reference image size
    
    properties
        % for density map calculation
        Vorocones = [];
        % average distance to neighbors expanded to map
        AvgDistanceMap = [];
        % cones per hex area (2*sqrt(3) * (("cone diameter") / 2)^ 2)
        % by defenition of Voronoi, we are getting here the "small" radius of hex
        DensityMatrixConesPerAreaNotFiltered = [];
        
        % distance between 2 cones in pixels
        % data points correspond to set of cones:
        %   conelocs = unique(hobj.Vorocones, 'rows', 'stable');
        %   conelocs(conelocs(:,1) < 0, :) = [];
        %   conelocs(conelocs(:,1) > imageWidth, :) = [];
        %   conelocs(conelocs(:,2) < 0, :) = [];
        %   conelocs(conelocs(:,2) > imageHeight, :) = [];
        AvgDistancesToNeighbors = [];
        

        % Fields defined in base class
%         ImageHeight = 0;
%         ImageWidth = 0;
        % cones per circle area after smoothing filter (gauss)
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
        function obj = NNDmean(coneLocs, imageSize)
            %NNDmean Construct an instance of this class
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
            
            Recalculate(obj);
        end
        
        function Recalculate(obj)
        %   Recalculate(obj) 
        %   recalculates all the data for density map.
        %   - obj - the current class object.
            
            [obj.AvgDistanceMap, obj.DensityMatrixConesPerAreaNotFiltered, obj.DensityMatrix, obj.AvgDistancesToNeighbors] = ...
                NNDmean.GetDensityMatrix(obj.Vorocones, obj.ImageHeight, obj.ImageWidth);
            
            [obj.PCD_cppa, obj.MinDensity_cppa, obj.PCD_loc] = NNDmean.GetMinMaxCPPA(obj.DensityMatrix);
            
            [obj.CDC20_density, obj.CDC20_loc, obj.Stats2] = NNDmean.GetCDC(obj.PCD_cppa, obj.DensityMatrix);
        end
        
        function s = saveobj(obj)
            % for density map calculation
            s.Vorocones = obj.Vorocones;
            s.ImageHeight = obj.ImageHeight;
            s.ImageWidth = obj.ImageWidth;
            s.DensityMatrix = obj.DensityMatrix;
            s.AvgDistanceMap = obj.AvgDistanceMap;
            s.DensityMatrixConesPerAreaNotFiltered = obj.DensityMatrixConesPerAreaNotFiltered;
            s.AvgDistancesToNeighbors = obj.AvgDistancesToNeighbors;

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
                newObj = NNDmean(); 
                % for density map calculation
                newObj.Vorocones = s.Vorocones;
                newObj.ImageHeight = s.ImageHeight;
                newObj.ImageWidth = s.ImageWidth;
                newObj.DensityMatrix = s.DensityMatrix;

                if isfield(s,'AvgDistanceMap')
                    newObj.AvgDistanceMap = s.AvgDistanceMap;
                end
                if isfield(s,'DensityMatrixConesPerAreaNotFiltered')
                    newObj.DensityMatrixConesPerAreaNotFiltered = s.DensityMatrixConesPerAreaNotFiltered;
                end
                if isfield(s,'AvgDistancesToNeighbors')
                    newObj.AvgDistancesToNeighbors = s.AvgDistancesToNeighbors;
                end

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
        
        function [densityMatrixRadial, densityMatrixArea, densityMatrixAreaFiltered, avgDistancesToNeighbors] = GetDensityMatrix(conelocs, imageHeight, imageWidth)
        %   densityMatrix = GetDensityMatrix(conelocs, imageHeight, imageWidth)
        %   returns a density matrix.
        %
        %   - conelocs - locations of cones.
        %   - imageHeight - height of the source image.
        %   - imageWidth - width of the source image.
        %
        % Returns:
        %   - densityMatrixRadial - the density matrix (imageHeight * imageWidth) with values, which represent cones per cone diameter (in pixels).
        %   - densityMatrixArea - the density matrix (imageHeight * imageWidth) with values, which represent cones per cone area (in pixels^2).
        %   - densityMatrixAreaFiltered - the density matrix (imageHeight *imageWidth)
        %   with values, which represent cones per cone area (in pixels^2), smoothed by gauss filter.
        %   - avgDistancesToNeighbors - values with avg distance to neighbor cones,
        %   correspoding to each cone.
            
            conelocs = unique(conelocs, 'rows', 'stable');
            % prepare data for voronoi plot
            conelocs(conelocs(:,1) < 0, :) = [];
            conelocs(conelocs(:,1) > imageWidth, :) = [];
            conelocs(conelocs(:,2) < 0, :) = [];
            conelocs(conelocs(:,2) > imageHeight, :) = [];

            boundingPoly = boundary(conelocs(:, 1), conelocs(:, 2), 1);
            
            vorocones = conelocs(:, 1:2);
            voronoiDelaunayTriang = delaunayTriangulation(vorocones);
            % V - verticies of polygons, C - the polygons constructed by V
            [~, C] = voronoiDiagram(voronoiDelaunayTriang);
            
            numberOfClosedPolygons = length(C);
            % Create a matrix from list of Verice inedexes (empty values filled by NaN)
            lengthesC = cellfun('length',C);
            maxLengthesC = max(lengthesC);
            % concat all points in one array from cell array of arrays
            allPolyConcateneted = cat(2, C{:});
            % preallocate memory for matrix
            matrixC = zeros([maxLengthesC + 1, numberOfClosedPolygons]);
            % find indexes where polygons definition ends
            indexes = sub2ind(size(matrixC), lengthesC(:).' + 1, 1:numberOfClosedPolygons);
            % fill it with NaN
            matrixC(indexes) =  1;
            matrixC = cumsum(matrixC(1:end - 1,:), 1);
            matrixC(matrixC == 1) = NaN;
            % fill polygons in the rest places of matrix
            matrixC(matrixC == 0) = allPolyConcateneted;
            matrixC = matrixC';
            
            waitbarHandler = waitbar(0, 'Density calc...');
            % find neighbors for all cones
            neightborLists = cell(numberOfClosedPolygons, 1);
            for indCone = 1:numberOfClosedPolygons
                waitbar(indCone / numberOfClosedPolygons, waitbarHandler);
                if any(boundingPoly == indCone)
                    continue;
                end
                
                polyVerticies = C{indCone};
                % we exclude vertex with [Inf, Inf] coords
                polyVerticies(polyVerticies == 1) = [];
                neighborsTemp = [];
                for indNeighbor = 1:length(polyVerticies)
                    % find neighbors with current vertex
                    [row, ~] = find(matrixC == polyVerticies(indNeighbor));
                    % exclude current cone
                    neighborsTemp = [neighborsTemp, row(row ~= indCone)'];
                end
                neightborLists{indCone} = unique(neighborsTemp);
            end
            
            % calc neighbor dists
            avgDistancesToNeighbors = zeros(numberOfClosedPolygons, 1);
            for indCone = 1:numberOfClosedPolygons
                avgDistancesToNeighbors(indCone) = mean(pdist2(vorocones(indCone, :), vorocones(neightborLists{indCone}, :)));
            end
            
            % interpolate the result to get a map
            densityMatrixRadial = NNDmean.InterpolateDensityMap(conelocs, avgDistancesToNeighbors, 'nearest', [imageHeight, imageWidth]);

            densityMatrixArea = (densityMatrixRadial) .^2 ./ 2 .* sqrt(3);
            densityMatrixArea = 1./densityMatrixArea;
            densityMatrixAreaFiltered = imgaussfilt(densityMatrixArea, 8);
            % invert matrix to represent it in the same way as other
            % densities
            
            close(waitbarHandler);
        end

        function map = InterpolateDensityMap(conelocs, baseDensityValues, methodInterpolation, imageSize)
        % map = InterpolateDensityMap(conelocs, baseDensityValues, methodInterpolation, imageSize)
        % interpolates whole density map by values, calculated in
        % conelocations.
        %   - conelocs - cone locations.
        %   - baseDensityValues - density values corresponding to conelocs.
        %   - methodInterpolation - method for scatteredInterpolant. Use 'nearest'.
        %   - imageSize - size of original image as [rows, cols].
        %
        % Returns: 
        %   - map - interpolated map, which will be look like voronoi patched blended
        %   together.

            minxc = ceil(min(conelocs(:,1)));
            maxxc = floor(max(conelocs(:,1)));
            maxyc = floor(max(conelocs(:,2)));
            minyc = ceil(min(conelocs(:,2)));
            % use the scatteredInterpolant to interpolate scattered data
            cdip = scatteredInterpolant(conelocs,baseDensityValues,methodInterpolation);
            % interpolate within the xaxv, yaxv area
            [xaxv, yaxv] = meshgrid(minxc:maxxc, minyc:maxyc);
            % execute interpolation
            cy = cdip(xaxv,yaxv);
            map = NaN(imageSize(1), imageSize(2));
            % fill data into a l matrix
            map(minyc:maxyc,minxc:maxxc) = cy;
        end
    end
end