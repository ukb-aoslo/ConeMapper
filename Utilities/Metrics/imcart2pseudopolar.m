function [ pseudoImage, maxRho ] = imcart2pseudopolar( sourceImage, rhoSampling, thetaSampling , location, method, rhostart )
%FUNCTION [ pseudoim ] = imcart2pseudopolar( im, rhoSampling, thetaSampling )
%   Robert Cooper
%
% This function takes in an image and converts it to pseudopolar, where
% Rho (the radius) is as long as half of the shortest side of the image,
% and Theta is sampled every other degree (by default)
%
% Change rhoUpsampling and thetaSampling to increase/decrease the sampling of the image.
% If you wish to upsample, then lower the *Sampling.
% If you wish to downsample, then raise the *Sampling.

    if ~exist('rhostart','var') || isempty(rhostart)
         rhostart = 0;
    end

    if ~exist('rhoSampling','var') || isempty(rhoSampling)
         rhoSampling = 1;
    end

    if ~exist('thetaSampling','var') || isempty(thetaSampling)
         thetaSampling = 1;
    end

    if ~exist('method','var') || isempty(method)
         method = 'linear';
    end

    if ~exist('location','var') || isempty(location)
         location = [floor(size(sourceImage,2)/2) + 1, floor(size(sourceImage,1)/2) + 1];
    end

    sourceImage = double(sourceImage);
    sourceImage(isnan(sourceImage)) = 0;

    [X, Y] = meshgrid( 1:size(sourceImage,2), 1:size(sourceImage,1) );

    rhoEnd = floor(min(size(sourceImage))/2) - 1;
    rho = rhostart:rhoSampling:rhoEnd;
    thetaStep = thetaSampling * 2*pi / 360;
    thetaEnd = 2*pi - thetaStep;
    theta = 0:thetaStep:thetaEnd;

    [R,T] = meshgrid(rho,theta);

    [Rx, Ty] = pol2cart(T,R);

    Rx = Rx + location(1);
    Ty = Ty + location(2);

    pseudoImage = interp2(X, Y, sourceImage, Rx, Ty, method);

    pseudoImage(isnan(pseudoImage)) = 0;
    maxRho = max(rho);
end