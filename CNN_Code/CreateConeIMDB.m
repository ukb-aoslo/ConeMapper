% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


function CreateConeIMDB(params)
% Function to create database file of cone and non cone-patches
% Data set sholud be either 'split detector' or 'confocal' to use the
% packaged sets, or add your own DataSet



%%%%%% Initialize Parameters 

% Get half patch size
HalfPatchSize = ceil((params.PatchSize-1)./2);

% load in list of images
ImageList = dir(fullfile( params.ImageDirTrain,['*' params.ImageExt '*'])); 
ImageList =  {ImageList.name};



%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Extract Patches %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numFiles = length(ImageList);
% AllClassLabels = tall(zeros(1, 0));
% AllPatches = tall(zeros(params.PatchSize, params.PatchSize, 1, 0));
AllClassLabels = zeros(1, 0, "int8");
AllPatches = zeros(params.PatchSize, params.PatchSize, 1, 0, "uint8");

wbHander = waitbar(0, "Assembling patches...");
% Loop through each file, and extract patches from each
for iFile = 1:numFiles

% Load Image
Image = imread(fullfile(params.ImageDirTrain,ImageList{iFile}));

% Perform preprocessing
Image = normalizeValues(Image,0,255);
    
% Load in manual coordinates (depending on type- change if using different formatting)
[~,BaseName] = fileparts(ImageList{iFile});
CoordPath = fullfile(params.ManualCoordDirTrain,[BaseName,params.CoordAdditionalText,params.CoordExt]);

switch params.CoordExt
    case '.csv'
        testRead = readmatrix(CoordPath);
        if isempty(testRead)
            ManualPos = zeros(0, 2);
        else
            ManualPos = round(csvread(CoordPath));
        end
    case '.txt'
        testRead = readmatrix(CoordPath);

        ManualPos = round(testRead);
%         [x,y] = textread(CoordPath);
%         ManualPos = [x,y];
%         ManualPos = round(ManualPos);
    otherwise
        error('Please select a known coord extension')     
end

% Remove manual markings too close to edge
Invalid = zeros(size(ManualPos,1),1);
if ~isempty(Invalid)
Invalid(ManualPos(:,1)<(1+HalfPatchSize)) = 1;
Invalid(ManualPos(:,1)>(size(Image,2)-HalfPatchSize)) = 1;
Invalid(ManualPos(:,2)<(1+HalfPatchSize)) = 1;
Invalid(ManualPos(:,2)>(size(Image,1)-HalfPatchSize)) = 1;

ManualPos(Invalid==1,:) = [];
end


%%% Voroni analysis to choose non-cone patches
% Get voronoi cells
VoronoiEdges = [];

if ~isempty(ManualPos) && (length(ManualPos(:, 1)) > 2)
    ManualPos = unique(ManualPos, 'rows');
    try
        [vx, vy] = voronoi(ManualPos(:,1),ManualPos(:,2));
        
    
        % Choose random point on each cell edge
        RandWeight = rand(1,size(vx,2));
        VoronoiEdges(:,1) = vx(1,:).*RandWeight + vx(2,:).*(1-RandWeight);
        VoronoiEdges(:,2) = vy(1,:).*RandWeight + vy(2,:).*(1-RandWeight);
        VoronoiEdges = round(VoronoiEdges);
    
        % Remove cases to close to the image boundary
        Invalid = zeros(size(VoronoiEdges,1),1);
        Invalid(VoronoiEdges(:,1)<(1+HalfPatchSize)) = 1;
        Invalid(VoronoiEdges(:,1)>(size(Image,2)-HalfPatchSize)) = 1;
        Invalid(VoronoiEdges(:,2)<(1+HalfPatchSize)) = 1;
        Invalid(VoronoiEdges(:,2)>(size(Image,1)-HalfPatchSize)) = 1;
    
        VoronoiEdges(Invalid==1,:) = [];
    catch ME
        disp(ME)
        disp(['filenum: ', num2str(iFile)]);
        disp(fullfile(params.ImageDirTrain,ImageList{iFile}));
        
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Extract and save patches along with label %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Loop Through and Extract patches for cones and non-cones
numManual = size(ManualPos,1);
numNonCone = size(VoronoiEdges,1);
Patches = zeros(params.PatchSize,params.PatchSize,numManual+numNonCone);
ClassLabels = cat(2,ones(1,numManual),2*ones(1,numNonCone));


% Get all cone patches
for iPatch = 1:numManual
    Patches(:,:,iPatch) = Image(ManualPos(iPatch,2)-HalfPatchSize:ManualPos(iPatch,2)+HalfPatchSize,ManualPos(iPatch,1)-HalfPatchSize:ManualPos(iPatch,1)+HalfPatchSize);
end
% Get non- cone patches
for iPatch = 1:numNonCone
    Patches(:,:,iPatch+numManual) = Image(VoronoiEdges(iPatch,2)-HalfPatchSize:VoronoiEdges(iPatch,2)+HalfPatchSize,VoronoiEdges(iPatch,1)-HalfPatchSize:VoronoiEdges(iPatch,1)+HalfPatchSize);
end

[h1, w1, z1] = size(Patches);
Patches = cast(reshape(Patches, h1, w1, 1, z1), "uint8");
% Patches = tall(Patches);
% ClassLabels = tall(ClassLabels);
% Save Patches and labels
ClassLabels = cast(ClassLabels, "int8");
AllPatches = cat(4,AllPatches,Patches);
AllClassLabels = cat(2,AllClassLabels,ClassLabels);

waitbar(iFile/numFiles, wbHander, "Assembling patches...");
end

% AllPatches = single(uint8(AllPatches));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Create and Save IMDB File %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Format the data
images.labels = single(AllClassLabels);
images.labels = AllClassLabels;
images.data = AllPatches; %reshape(AllPatches, size(AllPatches,1),size(AllPatches,2),1,size(AllPatches,3));
images.set = ones(size(images.labels), "int8");

% Set meta data
sets{1} = 'train';
sets{2} = 'val';
sets{3} = 'test';
classes{1} = {1};
classes{2} = {2};
meta.sets = sets;
meta.classes = classes;

%Save the data
save (params.imdb.SavePath,'-v7.3','images','meta')
