% Please cite this paper if you use any component of this software:
% D. Cunefare, L. Fang, R.F. Cooper, A. Dubra, J. Carroll, S. Farsiu, "Open source software for automatic detection of cone photoreceptors in adaptive optics ophthalmoscopy using convolutional neural networks," Scientific Reports, 7, 6620, 2017.
% Released under a GPL v2 license.


function [CNNPos, varargout] = GetConePosSingle(params,Image,net,ProbParam, cnnCalcType)
% Function to find cone positions using pretrained network and optimization
% parameters

 % load in the Net if nothing was passed
if nargin<3
    load(params.ProbMap.NetworkPath)
    net = vl_simplenn_move(net, cnnCalcType);
    net.layers{end}.type = 'softmax';
end

% Set detection parameters based on optimization if nothing was passed
if nargin<4
    load(params.Results.OptimizationPath)
    ProbParam.PMsigma = OptParam.MaxSigma;
    ProbParam.PMthresh = OptParam.MaxPMthresh;
    ProbParam.ExtMaxH = OptParam.MaxExtMaxH;
end

    
% Get half patch size
HalfPatchSize = ceil((params.PatchSize-1)./2);


% Perform preprocessing
    Image = normalizeValues(Image,0,255);
    
    % Padimage    
    PadImage = padarray(Image,[HalfPatchSize HalfPatchSize],'symmetric');
    
    % Get patches
    [test_patches] = im2patches(PadImage,[params.PatchSize  params.PatchSize],params.ProbMap.PatchDistance);
    
    % Resize patches to be same as used for network
    test_patches = single(reshape(test_patches, size(test_patches,1),size(test_patches,2),1,size(test_patches,3)));

    % Use CNN to find probability for each patch 
    NumPatches = size(test_patches,4);
    Test_Probability = zeros(2, NumPatches);
    
    isGpuUsed = 1;
    if strcmp(cnnCalcType, 'cpu')
        isGpuUsed = 0;
    end
    
    for Iter_num = 1:params.ProbMap.batchsize:NumPatches
        batchStart = Iter_num ;
        batchEnd = min(Iter_num+params.ProbMap.batchsize-1,NumPatches) ;
        batch = batchStart : 1 : batchEnd ;
        if isGpuUsed
            resized_test_patches = gpuArray(single(test_patches(:,:,:,batch)));
        else
            resized_test_patches = single(test_patches(:,:,:,batch));
        end
        res_temp = vl_simplenn(net, resized_test_patches,[],[],'mode','test');
        Prob_temp = squeeze(gather(res_temp(end).x)) ;
        Prob_temp(3:end,:) = [];
        Test_Probability(:, batch) = Prob_temp;
    end
    
    % Get Probability of being a cone
    Cone_Probability = Test_Probability(1,:);
    Cone_Probability = reshape(Cone_Probability',size(Image));
    
    
    % Determine cone locations
    [CNNPos] = ProbabilityMap_ConeLocations(Cone_Probability,ProbParam);
    
    if nargout > 1
        varargout = {};
        varargout{end + 1} = Cone_Probability;
    end