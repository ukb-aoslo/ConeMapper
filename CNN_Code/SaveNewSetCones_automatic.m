% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


function [conelocs] = SaveNewSetCones_automatic(params,ImageDir,ImExtension,boxposition,I, cnnCalcType)
% function to find cones using pretrained CNN on new images

% Get half patch size
% HalfPatchSize = ceil((params.PatchSize-1)./2);

% load in the Net (loads 'net' and 'stats')
load(params.ProbMap.NetworkPath)

disp(strcat('Move CNN to:  ', cnnCalcType, '. (can be up to 10 mins)'));
net = vl_simplenn_move(net, cnnCalcType);
net.layers{end}.type = 'softmax';

disp('start cone detection');

% Set detection parameters based on optimization
load(params.Results.OptimizationPath)

ProbParam.PMsigma = OptParam.MaxSigma;
ProbParam.PMthresh = OptParam.MaxPMthresh;
ProbParam.ExtMaxH = OptParam.MaxExtMaxH;

originalSizeX = size(I,2);
originalSizeY = size(I,1);
newSize = 150;

cutoutsINrowX = ceil(originalSizeX/newSize);      % calculate minimum numer of cutouts to produce
cutoutsINrowY = ceil(originalSizeY/newSize);

offsetX = floor(cutoutsINrowX*newSize-originalSizeX)/(cutoutsINrowX-1);

while offsetX < 20
    cutoutsINrowX = cutoutsINrowX+1;
    offsetX = round((cutoutsINrowX*newSize-originalSizeX)/(cutoutsINrowX-1));      % offset which will be applied for a fix number of cutouts in a row/column
end

offsetY = floor(cutoutsINrowY*newSize-originalSizeY)/(cutoutsINrowY-1);

while offsetY < 20
    cutoutsINrowY = cutoutsINrowY+1;
    offsetY = round((cutoutsINrowY*newSize-originalSizeY)/(cutoutsINrowY-1));      % offset which will be applied for a fix number of cutouts in a row/column
end

numCutouts = cutoutsINrowX*cutoutsINrowY;
Cutout = 1;
conelocs = [];

positions_startX = [1 newSize-offsetX:newSize-1-offsetX:((newSize-1)*cutoutsINrowX-offsetX*cutoutsINrowX)];
positions_endX = positions_startX+(newSize-1);

positions_startY = [1 newSize-offsetY:newSize-1-offsetY:((newSize-1)*cutoutsINrowY-offsetY*cutoutsINrowY)];
positions_endY = positions_startY+(newSize-1);

h = waitbar(0,'CNN locations...');

halfOffsetX = floor(offsetX/2);
halfOffsetY = floor(offsetY/2);
probMapFull = zeros(originalSizeY, originalSizeX);
probMaps = cell(cutoutsINrowY, cutoutsINrowX);

for y_cutout = 1:cutoutsINrowY
    for x_cutout = 1:cutoutsINrowX
         
        % create patches of 150x150 pixel
        x_start = positions_startX(x_cutout);
        y_start = positions_startY(y_cutout);
        x_end = positions_endX(x_cutout);
        y_end = positions_endY(y_cutout);
        
        cutout150 = I(y_start:y_end,x_start:x_end);
        Image = mat2gray(cutout150);
        %         imageSize = size(cutout150);

        % Get the cone positions;
        [CNNPos, probMap]= GetConePosSingle(params,Image,net,ProbParam, cnnCalcType);
        
        % skip empty CNNPos 
        if ~isempty(CNNPos)
            if x_start ~= 1
                CNNPos(CNNPos(:,1)<=halfOffsetX,:)         = [];
            end
            if x_end + halfOffsetX < originalSizeX
                CNNPos(CNNPos(:,1)>=newSize-halfOffsetX,:) = [];
            end
            if y_start ~= 1
                CNNPos(CNNPos(:,2)<=halfOffsetY,:)         = [];
            end
            if y_end + halfOffsetY < originalSizeY
                CNNPos(CNNPos(:,2)>=newSize-halfOffsetY,:) = [];
            end
            
            correct_xy = [ones(size(CNNPos,1),1)*(x_start-1), ones(size(CNNPos,1),1)*(y_start-1)];
            
            CNNPos_I = CNNPos+correct_xy;
            
            conelocs = [conelocs; CNNPos_I];
        end
        
        probMaps{y_cutout, x_cutout} = probMap;
        
        probMapFull(y_start:y_end,x_start:x_end) = probMap;
        
        Cutout = Cutout+1;
        
        waitbar(((y_cutout - 1) * cutoutsINrowX + x_cutout) / (numCutouts))
    end
end

close(h)

multiple_mosaics = 0;

% % Make the folder for saving data
% if(~exist(SaveDir,'dir'))
% mkdir(SaveDir);
% end

newMap = RecollectMap(probMaps, ...
        positions_startX, positions_startY, positions_endX, positions_endY, ...
        cutoutsINrowX, cutoutsINrowY, originalSizeY, originalSizeX, ...
        offsetX, offsetY);

% now we need to try to extract new cones from new map
% Determine cone locations
[CNNPos2] = ProbabilityMap_ConeLocations(newMap,ProbParam);
% conelocs = CNNPos2;

% [~,BaseName] = fileparts(ImageList{iFile});
imageSize = size(I);
SaveName = [ImageDir(1:end-4) '_annotated.mat'];
save(fullfile(SaveName),'I', 'boxposition', 'conelocs','multiple_mosaics','imageSize');
imwrite(probMapFull, [ImageDir(1:end-4), '_probMap.png']);

imwrite(newMap, [ImageDir(1:end-4), '_probMap2.png']);
end



function fullMap = RecollectMap(pieces, ...
        positionsStartX, positionsStartY, positionsEndX, positionsEndY, ...
        cutoutsINrowX, cutoutsINrowY, originalSizeY, originalSizeX, ...
        offsetX, offsetY)
    fullMap = zeros(originalSizeY, originalSizeX);
    
    % recollect middle part
    for yCutout = 1:cutoutsINrowY-1
        for xCutout = 1:cutoutsINrowX-1
            xStart = positionsStartX(xCutout);
            yStart = positionsStartY(yCutout);
            xEnd = positionsEndX(xCutout);
            yEnd = positionsEndY(yCutout);
            
            cutoutLeftTop = pieces{yCutout, xCutout};
            cutoutRightTop = pieces{yCutout, xCutout + 1};
            cutoutLeftBottom = pieces{yCutout + 1, xCutout};
            cutoutRightBottom = pieces{yCutout, xCutout + 1};
            
            doubleLineTop = cutoutLeftBottom(1:offsetY, offsetX+1:end-offsetX);
            doubleLineBottom = cutoutLeftTop(end-offsetY+1:end, offsetX+1:end-offsetX);
            doubleLineLeft = cutoutRightTop(offsetY+1:end-offsetY, 1:offsetX);
            doubleLineRight = cutoutLeftTop(offsetY+1:end-offsetY, end-offsetX+1:end);
            
            quaterSquare1 = cutoutLeftTop(end-offsetY+1:end, end-offsetX+1:end);
            quaterSquare2 = cutoutRightTop(end-offsetY+1:end, 1:offsetX);
            quaterSquare3 = cutoutLeftBottom(1:offsetY, end-offsetX+1:end);
            quaterSquare4 = cutoutRightBottom(1:offsetY, 1:offsetX);
            
            quaterAvg = (quaterSquare1 + quaterSquare2 + quaterSquare3 + quaterSquare4) ./ 4;
            bottomLineAvg = (doubleLineTop + doubleLineBottom) ./ 2;
            rightLineAvg = (doubleLineLeft + doubleLineRight) ./ 2;
            
            fullMap(yStart+offsetY:yEnd-offsetY, xStart+offsetX:xEnd-offsetX) = cutoutLeftTop(offsetY+1:end-offsetY, offsetX+1:end-offsetX);
            fullMap(yEnd-offsetY+1:yEnd, xStart+offsetX:xEnd-offsetX) = bottomLineAvg;
            fullMap(yStart+offsetY:yEnd-offsetY, xEnd-offsetX+1:xEnd) = rightLineAvg;
            fullMap(yEnd-offsetY+1:yEnd, xEnd-offsetX+1:xEnd) = quaterAvg;
        end
    end
    
    % recollect top and bottom rows
    for xCutout = 1:cutoutsINrowX-1
        xStart = positionsStartX(xCutout);
        xEnd = positionsEndX(xCutout);
            
        % top line
        yStart = positionsStartY(1);
        
        cutoutLeft = pieces{1, xCutout};
        cutoutRight = pieces{1, xCutout + 1};
        
        topLine = cutoutLeft(1:offsetY, offsetX+1:end-offsetX);
        quaterSquare1 = cutoutLeft(1:offsetY, end-offsetX+1:end);
        quaterSquare2 = cutoutRight(1:offsetY, 1:offsetX);
        
        quaterAvg = (quaterSquare1 + quaterSquare2) ./ 2;
        fullMap(yStart:yStart+offsetY-1, xStart+offsetX:xEnd-offsetX) = topLine;
        fullMap(yStart:yStart+offsetY-1, xEnd-offsetX+1:xEnd) = quaterAvg;
        
        % bottom line
        yStart = positionsStartY(cutoutsINrowY);
        yEnd = positionsEndY(cutoutsINrowY);
        
        cutoutLeft = pieces{cutoutsINrowY, xCutout};
        cutoutRight = pieces{cutoutsINrowY, xCutout + 1};
        
        bottomLine = cutoutLeft(end-offsetY+1:end, offsetX+1:end-offsetX);
        rightLine = cutoutLeft(offsetY+1:end, end-offsetX+1:end);
        leftLine = cutoutRight(offsetY+1:end, 1:offsetX);
        
        rightAvg = (rightLine + leftLine) ./ 2;
        fullMap(yStart+offsetY:yEnd-offsetY, xStart+offsetX:xEnd-offsetX) = cutoutLeft(offsetY+1:end-offsetY, offsetX+1:end-offsetX);
        fullMap(yEnd-offsetY+1:yEnd, xStart+offsetX:xEnd-offsetX) = bottomLine;
        fullMap(yStart+offsetY:yEnd, xEnd-offsetX+1:xEnd) = rightAvg;
    end
    
    % recollect left and right columns
    for yCutout = 1:cutoutsINrowY-1
        yStart = positionsStartY(yCutout);
        yEnd = positionsEndY(yCutout);
            
        % left line
        xStart = positionsStartX(1);
        
        cutoutTop = pieces{yCutout, 1};
        cutoutBottom = pieces{yCutout + 1, 1};
        
        leftLine = cutoutTop(offsetY+1:end-offsetY, 1:offsetX);
        quaterSquare1 = cutoutTop(end-offsetY+1:end, 1:offsetX);
        quaterSquare2 = cutoutBottom(1:offsetY, 1:offsetX);
        
        quaterAvg = (quaterSquare1 + quaterSquare2) ./ 2;
        fullMap(yStart+offsetY:yEnd-offsetY, xStart:xStart+offsetX-1) = leftLine;
        fullMap(yEnd-offsetY+1:yEnd, xStart:xStart+offsetX-1) = quaterAvg;
        
        % right line
        xStart = positionsStartX(cutoutsINrowX);
        xEnd = positionsEndX(cutoutsINrowX);
        cutoutTop = pieces{yCutout, cutoutsINrowX};
        cutoutBottom = pieces{yCutout + 1, cutoutsINrowX};
        
        rightLine = cutoutTop(offsetY+1:end-offsetY, end-offsetX+1:end);
        bottomLine = cutoutTop(end-offsetY+1:end, offsetX+1:end);
        topLine = cutoutBottom(1:offsetY, offsetX+1:end);
        
        bottomAvg = (bottomLine + topLine) ./ 2;
        fullMap(yStart+offsetY:yEnd-offsetY, xStart+offsetX:xEnd-offsetX) = cutoutTop(offsetY+1:end-offsetY, offsetX+1:end-offsetX);
        fullMap(yStart+offsetY:yEnd-offsetY, xEnd-offsetX+1:xEnd) = rightLine;
        fullMap(yEnd-offsetY+1:yEnd, xStart+offsetX:xEnd) = bottomAvg;
    end
    
    % recollect 3 corners
    leftTopCutout = pieces{1, 1};
    fullMap(1:offsetY, 1:offsetX) = leftTopCutout(1:offsetY, 1:offsetX);
    
    yEnd = positionsEndY(cutoutsINrowY);
    leftBottomCutout = pieces{cutoutsINrowY, 1};
    fullMap(yEnd-offsetY+1:yEnd, 1:offsetX) = leftBottomCutout(end-offsetY+1:end, 1:offsetX);
    
    xEnd = positionsEndX(cutoutsINrowX);
    rightTopCutout = pieces{1, cutoutsINrowX};
    fullMap(1:offsetY, xEnd-offsetX+1:xEnd) = rightTopCutout(1:offsetY, end-offsetX+1:end);
    
    % recollect last bottom right piece
    yStart = positionsStartY(cutoutsINrowY);
    yEnd = positionsEndY(cutoutsINrowY);
    xStart = positionsStartX(cutoutsINrowX);
    xEnd = positionsEndX(cutoutsINrowX);
    rightBottomCutout = pieces{cutoutsINrowY, cutoutsINrowX};
    fullMap(yStart+offsetY:yEnd, xStart+offsetX:xEnd) = rightBottomCutout(offsetY+1:end, offsetX+1:end);
end
