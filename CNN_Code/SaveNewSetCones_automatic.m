% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


function [conelocs] = SaveNewSetCones_automatic(params,ImageDir,ImExtension,boxposition,I, cnnCalcType)
% function to find cones using pretrained CNN on new images

% Get half patch size
HalfPatchSize = ceil((params.PatchSize-1)./2);

% load in the Net
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

offsetX = (cutoutsINrowX*newSize-originalSizeX)/(cutoutsINrowX-1);

while offsetX < 20
    cutoutsINrowX = cutoutsINrowX+1;
    offsetX = round((cutoutsINrowX*newSize-originalSizeX)/(cutoutsINrowX-1));      % offset which will be applied for a fix number of cutouts in a row/column
end

offsetY = (cutoutsINrowY*newSize-originalSizeY)/(cutoutsINrowY-1);

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
        [CNNPos]= GetConePosSingle(params,Image,net,ProbParam, cnnCalcType);
        
        % skip empty CNNPos 
        if ~isempty(CNNPos)
            % кажется офсеты тут перепутаны местами
            CNNPos(CNNPos(:,1)<=floor(offsetY/2),:)         = [];
            CNNPos(CNNPos(:,1)>=newSize-floor(offsetY/2),:) = [];
            CNNPos(CNNPos(:,2)<=floor(offsetX/2),:)         = [];
            CNNPos(CNNPos(:,2)>=newSize-floor(offsetX/2),:) = [];
            
            correct_xy = [ones(size(CNNPos,1),1)*(x_start-1), ones(size(CNNPos,1),1)*(y_start-1)];
            
            CNNPos_I = CNNPos+correct_xy;
            
            conelocs = [conelocs; CNNPos_I];
        end
        
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


% [~,BaseName] = fileparts(ImageList{iFile});
imageSize = size(I);
SaveName = [ImageDir(1:end-4) '_annotated.mat'];
save(fullfile(SaveName),'I', 'boxposition', 'conelocs','multiple_mosaics','imageSize');





