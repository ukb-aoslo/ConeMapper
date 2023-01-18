function conelocs = RemoveNeighborConesInRadius(image, conelocs, distDifference)
%RemoveNeighborConesInRadius Deletes all cones which is to close to the neighbors.
    offset = 5;
    [rows, columns, ~] = size(image);
    
    y = 0;
    step = 150;
    
    cutoutsINrowX = ceil(columns / step) + 1;
    cutoutsINrowY = ceil(rows / step) + 1;
    numCutouts = cutoutsINrowX * cutoutsINrowY;
    stepCounter = 0;
    delDoubleWaitbar = waitbar(0, 'Deleting double cones...');
    
    % for each tile with step*step size
    while y <= rows
        x = 0;
        while x <= columns
            % take all cones from tile
            indexes = find(conelocs(:, 1) >= x & conelocs(:, 1) <= x + step + offset ...
                & conelocs(:, 2) >= y & conelocs(:, 2) <= y + step + offset);
            
            conelocsY = conelocs(indexes, 1);
            conelocsX = conelocs(indexes, 2);
            
            R_all = [conelocsX, conelocsY];
            % calculate distance matrix for every point pair
            dist = pdist2(R_all, R_all);
            
            % example
            % 0 0 0 0
            % d 0 0 0
            % d d 0 0
            % d d d 0
            dist(1:(1+size(dist,1)):end) = inf;
            dist(logical(triu(ones(size(dist))))) = inf;
            
            % find all indexes of cones with low distances
            L = dist < distDifference;
            L = any(L, 1);
            
            % Delete cones
            conelocs(indexes(L),:) = [];
            
            x = x + step;
            stepCounter = stepCounter + 1;
            waitbar(stepCounter / (numCutouts), delDoubleWaitbar);
        end
        y = y + step;
    end
    
    close(delDoubleWaitbar);
end

