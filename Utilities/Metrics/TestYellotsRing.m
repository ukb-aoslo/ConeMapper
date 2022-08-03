
im1 = imread('BAK8044R_2019_04_03_10_31_23_AOSLO_V003_stabilized_840.tif');
% im1 = imread('test.png');
% im1 = rgb2gray(im1);

maxPixelValue = max(im1(:));
scaleConstant = 255 / log(double(1 + maxPixelValue));
newImage = scaleConstant * log(double(im1 + 1));

% figure;
% subplot(1, 2, 1);
% imshow(im1);
% title('source');
% subplot(1, 2, 2);
% imshow(uint8(round(newImage)));
% title('logTransform');

[height, width, ~] = size(newImage);

croppedIm1 = newImage(floor(height/2-75):floor(height/2+75), floor(width/2-75):floor(width/2+75));
% croppedIm1 = histeq(croppedIm1);
croppedFft = fft2(croppedIm1);
absShiftedIm1 = (abs(fftshift(croppedFft)));
% absShiftedIm1(74:76, 74:76) = 0;
% divided = abs(absShiftedIm1)./max(abs(absShiftedIm1(:)));
% divided(divided<0.15) = 0;

absShiftedIm1 = imresize(absShiftedIm1, 5, 'bicubic');
absShiftedIm1 = abs(log(absShiftedIm1 .^ 2 +1));

fig = figure;
imToPrint2 = absShiftedIm1./max(abs(absShiftedIm1(:)));
% imToPrint2 = imresize(imToPrint, 5, 'bicubic');
imshow(imToPrint2);



indH = 1;
indW = 1;

% Create a logical image of a circle with specified
% diameter, center, and image size.
% First create the image.
imageSizeX = 755;
imageSizeY = 755;
[columnsInImage, rowsInImage] = meshgrid(1:imageSizeX, 1:imageSizeY);
% Next create the circle in the image.
centerX = 377;
centerY = 377;
radius = 150;
radius2 = 120;
circlePixels = (rowsInImage - centerY).^2 + (columnsInImage - centerX).^2 <= radius.^2 ...
    & (rowsInImage - centerY).^2 + (columnsInImage - centerX).^2 >= radius2.^2;
% circlePixels is a 2D "logical" array.
% Now, display it.
figure
imshow(circlePixels) ;
title('Binary image of a circle');


c = normxcorr2(circlePixels,imToPrint2);
[ypeak,xpeak] = find(c==max(c(:)));
peakV = c(ypeak, xpeak);
c(ypeak, xpeak) = -Inf; 
[ySecPeak, xSecPeak] = find(c==max(c(:)));
peak2V = c(ySecPeak, xSecPeak);
disp(peak2V/peakV);

yoffSet = ypeak-size(circlePixels,1);
xoffSet = xpeak-size(circlePixels,2);

% while indH <= height
%     if indH-75 < 1 || indH + 75-1 >height
%         indH = indH + 50;
%         continue;
%     end
%     
%     while indW <= width
%         if indW-75 < 1 || indW + 75-1 >width
%             indW = indW + 50;
%             continue;
%         end
%         
%         croppedIm1 = im1(indH-75:indH+74, indW-75:indW+74);
%         croppedIm1 = histeq(croppedIm1);
%         croppedFft = fft2(croppedIm1);
%         absShiftedIm1 = (abs(fftshift(croppedFft)));
%         absShiftedIm1(70:80, 70:80) = 0;
%         divided = abs(absShiftedIm1)./max(abs(absShiftedIm1(:)));
%         divided(divided<0.15) = 0;
%         imwrite(divided, ['./testImages/im',num2str(indH), '_', num2str(indW), '.tiff']);
%         
%         indW = indW + 50;
%     end
%     indH = indH + 50;
%     indW = 1;
% end