folder = 'E:\PhD\DATA\Training_Data_for_NEW_Training';
fileName = 'E:\PhD\DATA\Training_Data_for_NEW_Training\BAK1012L1_2020_06_18_09_13_07_AOSLO_788_V001_annotated_JLR.mat';
dirs = dir([folder, '\*.mat']);

delDoubleWaitbar = waitbar(0,'Check distance...');
files = {};
count = 0;
dists = zeros(1, length(dirs));
for iFile = 1:length(dirs)
    load([dirs(iFile).folder, '\', dirs(iFile).name], 'I', 'conelocs');
    offset = 5;
    if iscell(I)
        [rows, columns, ~] = size(I{1});
    else
        [rows, columns, ~] = size(I);
    end

    y = 0;
    step = 150;
    distDifference = 3;

    cutoutsINrowX = ceil(columns / step) + 1;
    cutoutsINrowY = ceil(rows / step) + 1;
    numCutouts = cutoutsINrowX * cutoutsINrowY;
    stepCounter = 0;
    conelocs(conelocs(:, 3) == 0, :) = [];

    minimumOld = Inf;
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

            minimumNew = min(dist, [], 'all');

            if minimumNew < minimumOld
                minimumOld = minimumNew;
            end

            x = x + step;
            stepCounter = stepCounter + 1;
        end
        y = y + step;
    end
    dists(iFile) = minimumOld;
%     disp(minimumOld);
    
    waitbar(iFile / (length(dirs)), delDoubleWaitbar);
    if minimumOld < 1
        count = count + 1;
        files{count} = [dirs(iFile).folder, '\', dirs(iFile).name];
    end
end

close(delDoubleWaitbar);