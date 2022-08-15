function [avgPixelSpac, interpedSpacMap, interpedConfMap, sumMap, imBox] = FitFourierSpacing(sourceImage, roiSize)
%FITFOURIERSPACING Summary of this function goes here
%   Robert Cooper
%
% - sourceImage - The image that will be analyzed. The only requirement 
%   is that it is a 2d, grayscale (1 channel) image.
% - roiSize - The side length (in pixels) of a sliding roi window- 
%   The roi will march along the image you've provided at a rate of 1/roi_step 
%   the size of the ROI, creating a "map" of spacing of the image.
% 
% #### Outputs:
% 
% - **avg_pixel_spac**: The average spacing of the image.
% - **interped_spac_map**: The spacing map of the input image (in pixel spacing).
% - **interped_conf_map**: The confidence map of the input image.
% - **sum_map**: The map corresponding to the amount of ROI overlap across the output map.
% - **imbox**: The bounding region of valid (nonzero, NaN, or Inf) pixels.

    superSampling = false;
    rowOrCell = 'cell';
    rdivider = 8;
    roiStep = floor(roiSize/rdivider);
    imComponents = bwconncomp(imclose(sourceImage > 0, ones(5)));
    imBox = regionprops(imComponents, 'BoundingBox');
    imSize = size(sourceImage);

    boxSizes = zeros(size(imBox, 1), 1);
    for indI = 1:size(imBox, 1)
        boxSizes(indI) = imBox(indI).BoundingBox(3) * imBox(indI).BoundingBox(4);
    end
    [~, maxSizeIndex] = max(boxSizes);
    imBox = floor(imBox(maxSizeIndex).BoundingBox);

    imBox(imBox <= 0) = 1;
    widthDiff = imSize(2) - (imBox(1) + imBox(3));
    if widthDiff < 0
        imBox(3) = imBox(3) + widthDiff;
    end
    
    heightDiff = imSize(1) - (imBox(2) + imBox(4));
    if heightDiff < 0 
        imBox(4) = imBox(4) + heightDiff;
    end
    
    if any(imSize(1:2) <= roiSize)
        % Our roi size should always be divisible by 2 (for simplicity).
        if rem(max(roiSize), 2) ~= 0
            roiSize = max(roiSize) - 1;
        end

        % If our image is oblong and smaller than a default roi_size, then pad
        % it to be as large as the largest edge.
        padSize = ceil((max(roiSize) - imSize) / 2);
        regionsOfInterest = {padarray(sourceImage, padSize, 'both')};
    else
        regionsOfInterest = cell(round((size(sourceImage) - roiSize) / roiStep));
        
        % Our roi size should always be divisible by 2 (for simplicity).
        if rem(roiSize, 2) ~= 0
            roiSize = roiSize - 1;
        end
        
        for indI = imBox(2):roiStep:imBox(2) + imBox(4) - roiSize
            for indJ = imBox(1):roiStep:imBox(1) + imBox(3) - roiSize
                numzeros = sum(sum(sourceImage(indI:indI + roiSize - 1, indJ:indJ + roiSize - 1) <= 10));

                if numzeros < (roiSize*roiSize)*0.05
                    regionsOfInterest{round(indI/roiStep)+1, round(indJ/roiStep)+1} = ...
                        sourceImage(indI:indI + roiSize - 1, indJ:indJ + roiSize - 1);
                else
                    regionsOfInterest{round(indI/roiStep)+1, round(indJ/roiStep)+1} = [];
                end
            end
        end 
    end
    
    pixelSpac = nan(size(regionsOfInterest));
    confidence = nan(size(regionsOfInterest));
    
    waitbarHandle = waitbar(0, 'Fourier fitting ...');
    lengthPixelSpac = length(pixelSpac(:));
    for indR = 1:lengthPixelSpac
        if ~isempty(regionsOfInterest{indR})    
            
            % We don't want this run on massive images (RAM sink)
            if superSampling == true
                % For reasoning, cite this: https://arxiv.org/pdf/1401.2636.pdf
                padSize = roiSize(1) * 6;
                padSize = (padSize - roiSize(1)) / 2 + 1;

                powerSpectrum = fftshift(fft2(padarray(regionsOfInterest{indR}, [padSize, padSize])));
                powerSpectrum = imresize(log10(abs(powerSpectrum).^2), [2048, 2048]);
                % Exclude the DC term from our radial average
                rhoStart = ceil(2048 / min(imSize));
            else
                % Make our hanning window for each ROI?
                hann_twodee = 1;
                powerSpectrum = fftshift(fft2(hann_twodee .* double(regionsOfInterest{indR})));
                powerSpectrum = log10(abs(powerSpectrum).^2);
                % Exclude the DC term from our radial average
                rhoStart = 1;
            end

            rhoSampling = 0.5;
            thetaSampling = 1;

            [polarRoi, powerSpectrumRadius] = imcart2pseudopolar(...
                powerSpectrum, rhoSampling, thetaSampling, [], 'makima' , rhoStart);
            polarRoi = circshift(polarRoi, -90 / thetaSampling, 1);

            upperLowerN = [thetaSampling:45, 136:225, 316:360] / thetaSampling;
            leftRightN = [46:135, 226:315] / thetaSampling;
            upperLowerNFourierProfile = mean(polarRoi(upperLowerN, :));
            leftRightNFourierProfile = mean(polarRoi(leftRightN, :));

            if strcmp(rowOrCell, 'cell') && ~all(isinf(leftRightNFourierProfile)) && ~all(isnan(leftRightNFourierProfile))
                [pixelSpac(indR), ~, confidence(indR)] = fourierFit(leftRightNFourierProfile, [], false);
                pixelSpac(indR) = 1 / (pixelSpac(indR) / ((powerSpectrumRadius*2) / rhoSampling));
            elseif strcmp(rowOrCell,'row') && ~all(isinf(upperLowerNFourierProfile)) && ~all(isnan(upperLowerNFourierProfile))
                [pixelSpac(indR), ~, confidence(indR)] = fourierFit(upperLowerNFourierProfile, [], false);
                pixelSpac(indR) = 1 / (pixelSpac(indR) / ((powerSpectrumRadius*2) / rhoSampling));
            else
                pixelSpac(indR) = NaN;
            end
        end
        
        waitbar(indR / lengthPixelSpac, waitbarHandle);
    end
    close(waitbarHandle);
    
    avgPixelSpac = mean(pixelSpac(~isnan(pixelSpac)));
    interpedSpacMap = avgPixelSpac;
    interpedConfMap = confidence;
    
    % If we've sampled over the region, then create the heat map
    if length(regionsOfInterest) > 1
        interpedSpacMap = zeros(imSize);
        interpedConfMap = zeros(imSize);
        sumMap = zeros(imSize);
        roiCoverage = roiSize;
        hann_twodee = 1;

        for indI = imBox(2):roiStep:imBox(2) + imBox(4) - roiSize
            for indJ = imBox(1):roiStep:imBox(1) + imBox(3) - roiSize
                if ~isnan(pixelSpac(round(indI/roiStep) + 1, round(indJ/roiStep) + 1))
                    thiserr = confidence(round(indI/roiStep) + 1, round(indJ/roiStep) + 1)^2;

                    interpedConfMap(indI:indI+roiCoverage-1, indJ:indJ+roiCoverage-1) = ...
                        interpedConfMap(indI:indI+roiCoverage-1, indJ:indJ+roiCoverage-1) +(hann_twodee*thiserr);
                    
                    thisspac = pixelSpac(round(indI/roiStep)+1,round(indJ/roiStep)+1);
                    interpedSpacMap(indI:indI+roiCoverage-1, indJ:indJ+roiCoverage-1) =...
                        interpedSpacMap(indI:indI+roiCoverage-1, indJ:indJ+roiCoverage-1) + (hann_twodee*(thiserr*thisspac));

                    sumMap(indI:indI+roiCoverage-1, indJ:indJ+roiCoverage-1) =...
                        sumMap(indI:indI+roiCoverage-1, indJ:indJ+roiCoverage-1) + hann_twodee;
                end
            end
        end

        interpedSpacMap = interpedSpacMap( imBox(2):imBox(2)+imBox(4), imBox(1):imBox(1)+imBox(3) );
        interpedConfMap = interpedConfMap( imBox(2):imBox(2)+imBox(4), imBox(1):imBox(1)+imBox(3) );
        sumMap = sumMap( imBox(2):imBox(2)+imBox(4), imBox(1):imBox(1)+imBox(3) );

        if nargout <= 1
            if strcmp(rowOrCell,'cell')
                imageToShow = interpedSpacMap ./ interpedConfMap;
            elseif strcmp(rowOrCell,'row')
                imageToShow = (2/sqrt(3)) .* interpedSpacMap ./ interpedConfMap;       
            end
            figure(1);
            clf;
            imagesc(imageToShow); 
            axis image;
            
            figure(2);
            clf; 
            imagesc((interpedConfMap ./ sumMap)); 
            colormap hot;
            
            figure(3);
            clf; 
            imagesc(sumMap); 
            axis image; 
            colormap gray;
        end
    end
end