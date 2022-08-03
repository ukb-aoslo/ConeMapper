classdef JuliusDensity < handle
    %JuliusDensity calculates the cone density using average of the
    %distances to the neighbor cones. Neighbor in terms of the Voronoi
    %diagram
    
    % requested input data
    % - locations of all cones which could be identified in the reference image
    % - reference image size
    
    properties
        % for density map calculation
        Vorocones = [];
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
        
        % points which represent a polygon inside of which we have non
        % aproximated map
        GoodPointsEdge = [];
    end
    
    methods
        function obj = JuliusDensity(coneLocs, imageSize, sourceImage)
            %JuliusDensity Construct an instance of this class
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
            
            if nargin < 3
                sourceImage = [];
            end
            
            Recalculate(obj, sourceImage);
        end
        
        function Recalculate(obj, sourceImage)
        %   Recalculate(obj, sourceImage) 
        %   recalculates all the data for density map.
        %   - obj - the current class object.            
            if nargin < 3
                sourceImage = [];
            end
            
            [obj.GoodPointsEdge, obj.DensityMatrix] = ...
                JuliusDensity.GetDensityMatrix(obj.Vorocones, obj.ImageHeight, obj.ImageWidth, sourceImage);
            
            [obj.PCD_cppa, obj.MinDensity_cppa, obj.PCD_loc] = JuliusDensity.GetMinMaxCPPA(obj.DensityMatrix);
            
            [obj.CDC20_density, obj.CDC20_loc, obj.Stats2] = JuliusDensity.GetCDC(obj.PCD_cppa, obj.MinDensity_cppa, obj.DensityMatrix);
        end
        
        function s = saveobj(obj)
            % for density map calculation
            s.Vorocones = obj.Vorocones;
            s.ImageHeight = obj.ImageHeight;
            s.ImageWidth = obj.ImageWidth;
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
        end
    end
    
    methods(Static)
        function obj = loadobj(s)
            if isstruct(s)
                newObj = JuliusDensity(); 
                % for density map calculation
                newObj.Vorocones = s.Vorocones;
                newObj.ImageHeight = s.ImageHeight;
                newObj.ImageWidth = s.ImageWidth;
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
                obj = newObj;
            else
                obj = s;
            end
        end
        
        function [goodPointsEdge, densityMatrix] = GetDensityMatrix(conelocs, imageHeight, imageWidth, sourceImage)
        %   densityMatrix = GetDensityMatrix(conelocs, imageHeight, imageWidth)
        %   returns a density matrix.
        %   - conelocs - locations of cones.
        %   - imageHeight - height of the source image.
        %   - imageWidth - width of the source image.
        %   - numOfNearestCones - number of cones in the area.
        %   - coneArea - array with area of each cone.
            
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
            [V,C] = voronoiDiagram(voronoiDelaunayTriang);
            
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
            [X,Y] = meshgrid(1:imageWidth, 1:imageHeight);
            X = reshape(X,[],1);
            Y = reshape(Y,[],1);
            Vq = griddata(floor(vorocones(:, 1)), floor(vorocones(:, 2)), avgDistancesToNeighbors, X, Y, 'cubic');
            Vq = reshape(Vq, imageHeight, imageWidth);
            densityMatrix = 1./Vq;
            
%             how to filter the result
            densityMatrix = sgolayfilt(1./Vq, 3, 11); 
            densityMatrix = sgolayfilt(densityMatrix' , 3, 11)'; 
%             imagesc(test');
            goodPointsEdge = [];
            
            close(waitbarHandler);
            % to plot Voronoi
            
            % create an alphaShape
%             voronoiAlphaShape = alphaShape(vorocones(:, 1), vorocones(:, 2));
%             voronoiAlphaShape.Alpha = 2 * voronoiAlphaShape.Alpha;
%             V(V(:, 1) == Inf | V(:, 2) == Inf, :) = -1;
%             % fill with NaN all verticies which is out of
%             % AlphaShape
%             tf = inShape(voronoiAlphaShape, V(:,1), V(:, 2));
%             V(~tf, 1) = NaN;
%             V(~tf, 2) = NaN;
%             
%             patch('Faces',matrixC,'Vertices',V,'FaceVertexCData',avgDistancesToNeighbors,'FaceColor','flat');
%             colorbar;
%             oldcmap = colormap;
%             colormap( flipud(oldcmap) );
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