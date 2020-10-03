function [ cnnFloderName, cnnCalcType, isCanceled ] = SelectVersionCNN()
%GETVERSIONCNN Summary of this function goes here
%   Detailed explanation goes here

fn = {'CNN v23 (gpu)', 'CNN v23 (cpu)', 'CNN v25 (gpu)', 'CNN v25 (cpu)'};
[indx,tf] = listdlg('Name', 'Cone Finder Algroithms', 'PromptString', {'Select a cone finder algorithm.'},...
    'SelectionMode','single','ListString', fn);

% if user clicked 'Cancel'
switch indx
    case 1 % 'CNN v23 (gpu)'
        cnnFloderName = 'matconvnet-1.0-beta23';
        cnnCalcType = 'gpu';
        
    case 2 % 'CNN v23 (cpu)'
        cnnFloderName = 'matconvnet-1.0-beta23cpu';
        cnnCalcType = 'cpu';
        
    case 3 % 'CNN v25 (gpu)'
        cnnFloderName = 'matconvnet-1.0-beta25';
        cnnCalcType = 'gpu';
        
    case 4 % 'CNN v25 (cpu)'
        cnnFloderName = 'matconvnet-1.0-beta25cpu';
        cnnCalcType = 'cpu';
end

isCanceled = ~tf;
end

