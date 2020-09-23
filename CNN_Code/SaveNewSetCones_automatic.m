% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


function [conelocs] = SaveNewSetCones_automatic(params,ImageDir,ImExtension,boxposition,I)
% function to find cones using pretrained CNN on new images

% Get half patch size
HalfPatchSize = ceil((params.PatchSize-1)./2);

% load in the Net
load(params.ProbMap.NetworkPath)

disp('Move CNN to GPU (can be up to 10 mins)');
net = vl_simplenn_move(net, 'gpu');
net.layers{end}.type = 'softmax';

disp('start cones recognision');

% Set detection parameters based on optimization
load(params.Results.OptimizationPath)

ProbParam.PMsigma = OptParam.MaxSigma;
ProbParam.PMthresh = OptParam.MaxPMthresh;
ProbParam.ExtMaxH = OptParam.MaxExtMaxH;

originalSize = size(I,1);
newSize = 150;

cutoutsINrow = ceil(originalSize/newSize);      % calculate minimum numer of cutouts to produce

offset = (cutoutsINrow*newSize-originalSize)/(cutoutsINrow-1);

while offset < 20
    cutoutsINrow = cutoutsINrow+1;
    offset = round((cutoutsINrow*newSize-originalSize)/(cutoutsINrow-1));      % offset which will be applied for a fix number of cutouts in a row/column
end

numCutouts = cutoutsINrow*cutoutsINrow;
Cutout = 1;
conelocs = [];

h = waitbar(0,'CNN locations...');
steps = cutoutsINrow;

for y_cutout = 1:cutoutsINrow
    for x_cutout = 1:cutoutsINrow
        
        % create patches of 150x150 pixel
        
        positions_start = [1 newSize-offset:newSize-1-offset:((newSize-1)*cutoutsINrow-offset*cutoutsINrow)];
        positions_end = positions_start+(newSize-1);
        
        x_start = positions_start(x_cutout);
        y_start = positions_start(y_cutout);
        x_end = positions_end(x_cutout);
        y_end = positions_end(y_cutout);
        
        cutout150 = I(y_start:y_end,x_start:x_end);
        Image = mat2gray(cutout150);
        %         imageSize = size(cutout150);
        
        
        % Get the cone positions;
        [CNNPos]= GetConePosSingle(params,Image,net,ProbParam);
        
        
        if isempty(CNNPos)
            % skip this cutout
        else
            CNNPos(CNNPos(:,1)<=floor(offset/2),:)         = [];
            CNNPos(CNNPos(:,1)>=newSize-floor(offset/2),:) = [];
            CNNPos(CNNPos(:,2)<=floor(offset/2),:)         = [];
            CNNPos(CNNPos(:,2)>=newSize-floor(offset/2),:) = [];
            
            correct_xy = [ones(size(CNNPos,1),1)*(x_start-1), ones(size(CNNPos,1),1)*(y_start-1)];
            
            CNNPos_I = CNNPos+correct_xy;
            %         CNNPos_I(:,1) = CNNPos(:,1)+correct_x;
            %         CNNPos_I(:,2) = CNNPos(:,2)+correct_y;
            
            conelocs = [conelocs; CNNPos_I];
        end
        
        Cutout = Cutout+1;
        
        waitbar(((y_cutout - 1) * cutoutsINrow + x_cutout) / (numCutouts))
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





