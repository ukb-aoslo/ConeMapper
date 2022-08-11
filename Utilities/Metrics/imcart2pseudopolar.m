function [ pseudoim, maxrho ] = imcart2pseudopolar( im, rhoSampling, thetaSampling , location, method, rhostart )
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
     rhostart =0;
end

if ~exist('rhoSampling','var') || isempty(rhoSampling)
     rhoSampling =1;
end

if ~exist('thetaSampling','var') || isempty(thetaSampling)
     thetaSampling =1;
end

if ~exist('method','var') || isempty(method)
     method ='linear';
end

if ~exist('location','var') || isempty(location)
     location = [floor(size(im,2)/2)+1 floor(size(im,1)/2)+1];
end

im = double(im);
im(isnan(im)) = 0;
%%
[X, Y]= meshgrid( 1:size(im,2), 1:size(im,1) );

rho = rhostart:rhoSampling: floor(min(size(im))/2)-1;
theta_step = thetaSampling*2*pi/360;
theta = 0: theta_step: 2*pi-theta_step;

[R,T] = meshgrid(rho,theta);

[Rx, Ty] = pol2cart(T,R);

Rx = Rx + location(1);
Ty = Ty + location(2);

pseudoim = interp2(X,Y,im,Rx,Ty,method);

pseudoim(isnan(pseudoim)) = 0;
maxrho = max(rho);
% imagesc( pseudoim ); colormap gray; axis image;
end

